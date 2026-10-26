"""Audit CLI comparing relevance-blind and existing hinted teacher scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from .model_score_chain_v1 import verify_model_score_manifest
from .question_set_v1 import read_jsonl, write_jsonl
from .relevance_blind_model_score_chain_v1 import verify_relevance_blind_model_score_manifest
from .teacher_chain_v1 import file_descriptor, file_sha256

AUDIT_SCHEMA_VERSION = "relevance-blind-vs-hinted-score-audit-v1"
AUDIT_MANIFEST_FILENAME = "relevance_blind_vs_hinted_audit_manifest.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _label_pairs(rows: Iterable[dict[str, Any]], episode_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        episode_id, slot = row.get("episode_id"), row.get("teacher_slot")
        if episode_id not in episode_ids or slot not in {"a", "b"} or slot in pairs[episode_id]:
            raise ValueError("teacher score labels do not form unique a/b pairs for the expected Episodes")
        pairs[episode_id][slot] = row
    if set(pairs) != set(episode_ids) or any(set(pair) != {"a", "b"} for pair in pairs.values()):
        raise ValueError("teacher score labels do not contain exactly two slots for every Episode")
    return dict(pairs)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _axis_row(rows: list[dict[str, Any]], *, axis: str, key: str) -> dict[str, Any]:
    deltas = [float(row["mean_score_delta_blind_minus_hinted"]) for row in rows]
    abs_deltas = [abs(value) for value in deltas]
    jointly_supervised = [row for row in rows if row["blind_supervised"] and row["hinted_supervised"]]
    target_deltas = [float(row["target_delta_blind_minus_hinted"]) for row in jointly_supervised]
    return {
        "schema_version": f"relevance-blind-vs-hinted-by-{axis}-v1",
        key: rows[0][key],
        "dimension_count": len(rows),
        "mean_blind_score": fmean(float(row["blind_mean_score_0_4"]) for row in rows),
        "mean_hinted_score": fmean(float(row["hinted_mean_score_0_4"]) for row in rows),
        "mean_score_delta_blind_minus_hinted": fmean(deltas),
        "mean_absolute_score_delta": fmean(abs_deltas),
        "max_absolute_score_delta": max(abs_deltas),
        "blind_higher_count": sum(value > 0 for value in deltas),
        "equal_mean_score_count": sum(value == 0 for value in deltas),
        "hinted_higher_count": sum(value < 0 for value in deltas),
        "blind_supervised_count": sum(bool(row["blind_supervised"]) for row in rows),
        "hinted_supervised_count": sum(bool(row["hinted_supervised"]) for row in rows),
        "jointly_supervised_count": len(jointly_supervised),
        "mask_agreement_count": sum(row["blind_supervised"] == row["hinted_supervised"] for row in rows),
        "mask_agreement_rate": sum(row["blind_supervised"] == row["hinted_supervised"] for row in rows) / len(rows),
        "mean_target_delta_blind_minus_hinted": _mean(target_deltas),
    }


def build_relevance_blind_vs_hinted_audit(*, dataset_root: Path, blind_manifest_path: Path, hinted_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    blind_manifest, blind_paths, blind_verified = verify_relevance_blind_model_score_manifest(
        blind_manifest_path, dataset_root=dataset_root,
    )
    hinted_manifest, hinted_paths, hinted_verified = verify_model_score_manifest(
        hinted_manifest_path, dataset_root=dataset_root,
        expected_aggregate_path=Path(hinted_manifest_path).parent / "aggregated_model_scores.jsonl",
    )
    if blind_manifest["source_chain"]["sha256"] != hinted_manifest["source_chain"]["sha256"]:
        raise ValueError("blind and hinted score chains must bind the same source teacher-chain manifest")
    if blind_verified["episode_ids"] != hinted_verified["episode_ids"]:
        raise ValueError("blind and hinted score chains must cover the same Episodes")
    if blind_verified["model_ids"] != hinted_verified["model_ids"]:
        raise ValueError("blind and hinted score chains must use the same model order")
    episode_ids = blind_verified["episode_ids"]
    model_ids = blind_verified["model_ids"]
    blind_pairs = _label_pairs(read_jsonl(blind_paths["teacher_scores"]), episode_ids)
    hinted_pairs = _label_pairs(read_jsonl(hinted_paths["teacher_scores"]), episode_ids)
    blind_threshold = float(blind_manifest["max_score_gap_for_supervision"])
    hinted_threshold = float(hinted_manifest["max_score_gap_for_supervision"])

    dimensions: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        for model_id in model_ids:
            blind_scores = [float(blind_pairs[episode_id][slot]["model_scores"][model_id]) for slot in ("a", "b")]
            hinted_scores = [float(hinted_pairs[episode_id][slot]["model_scores"][model_id]) for slot in ("a", "b")]
            blind_mean, hinted_mean = fmean(blind_scores), fmean(hinted_scores)
            blind_gap, hinted_gap = abs(blind_scores[0] - blind_scores[1]), abs(hinted_scores[0] - hinted_scores[1])
            blind_supervised, hinted_supervised = blind_gap <= blind_threshold, hinted_gap <= hinted_threshold
            dimensions.append({
                "schema_version": "relevance-blind-vs-hinted-dimension-v1",
                "episode_id": episode_id, "model_id": model_id,
                "blind_teacher_scores": blind_scores, "hinted_teacher_scores": hinted_scores,
                "blind_mean_score_0_4": blind_mean, "hinted_mean_score_0_4": hinted_mean,
                "mean_score_delta_blind_minus_hinted": blind_mean - hinted_mean,
                "blind_score_gap": blind_gap, "hinted_score_gap": hinted_gap,
                "blind_supervised": blind_supervised, "hinted_supervised": hinted_supervised,
                "mask_changed": blind_supervised != hinted_supervised,
                "blind_target_0_1": blind_mean / 4.0 if blind_supervised else None,
                "hinted_target_0_1": hinted_mean / 4.0 if hinted_supervised else None,
                "target_delta_blind_minus_hinted": (blind_mean - hinted_mean) / 4.0 if blind_supervised and hinted_supervised else None,
            })

    by_model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dimensions:
        by_model_groups[row["model_id"]].append(row)
        by_episode_groups[row["episode_id"]].append(row)
    by_model = [_axis_row(by_model_groups[model_id], axis="model", key="model_id") for model_id in model_ids]
    by_episode = [_axis_row(by_episode_groups[episode_id], axis="episode", key="episode_id") for episode_id in episode_ids]
    deltas = [float(row["mean_score_delta_blind_minus_hinted"]) for row in dimensions]
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "source_manifest_sha256": blind_manifest["source_chain"]["sha256"],
        "episode_count": len(episode_ids), "model_count": len(model_ids),
        "dimension_count": len(dimensions),
        "mean_score_delta_blind_minus_hinted": fmean(deltas),
        "mean_absolute_score_delta": fmean(abs(value) for value in deltas),
        "mask_change_count": sum(bool(row["mask_changed"]) for row in dimensions),
        "mask_change_rate": sum(bool(row["mask_changed"]) for row in dimensions) / len(dimensions),
        "blind_supervised_dimensions": sum(bool(row["blind_supervised"]) for row in dimensions),
        "hinted_supervised_dimensions": sum(bool(row["hinted_supervised"]) for row in dimensions),
        "blind_prompt_version": blind_manifest["prompt_version"],
        "hinted_prompt_version": hinted_manifest["prompt_version"],
        "blind_teacher_identities": blind_manifest["teacher_identities"],
        "hinted_teacher_identities": hinted_manifest["teacher_identities"],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "dimension_rows": output_dir / "relevance_blind_vs_hinted_dimensions.jsonl",
        "by_model_rows": output_dir / "relevance_blind_vs_hinted_by_model.jsonl",
        "by_episode_rows": output_dir / "relevance_blind_vs_hinted_by_episode.jsonl",
        "summary": output_dir / "relevance_blind_vs_hinted_summary.json",
        "audit_manifest": output_dir / AUDIT_MANIFEST_FILENAME,
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("refusing to overwrite relevance-blind comparison audit artifacts")
    write_jsonl(output_paths["dimension_rows"], dimensions)
    write_jsonl(output_paths["by_model_rows"], by_model)
    write_jsonl(output_paths["by_episode_rows"], by_episode)
    _write_json(output_paths["summary"], summary)
    audit_manifest = {
        "schema_version": "relevance-blind-vs-hinted-score-audit-manifest-v1",
        "created_at": _timestamp(),
        "inputs": {
            "blind_manifest": {"path": str(Path(blind_manifest_path).resolve()), "bytes": Path(blind_manifest_path).stat().st_size, "sha256": file_sha256(Path(blind_manifest_path))},
            "hinted_manifest": {"path": str(Path(hinted_manifest_path).resolve()), "bytes": Path(hinted_manifest_path).stat().st_size, "sha256": file_sha256(Path(hinted_manifest_path))},
        },
        "outputs": {
            role: file_descriptor(path, relative_to=output_dir, rows=(len(dimensions) if role == "dimension_rows" else len(by_model) if role == "by_model_rows" else len(by_episode) if role == "by_episode_rows" else None))
            for role, path in output_paths.items() if role != "audit_manifest"
        },
        "episode_ids": episode_ids, "model_ids": model_ids,
        "summary_sha256": file_sha256(output_paths["summary"]),
    }
    _write_json(output_paths["audit_manifest"], audit_manifest)
    return {"summary": summary, "dimensions": dimensions, "by_model": by_model, "by_episode": by_episode, "paths": output_paths, "manifest": audit_manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--hinted-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_relevance_blind_vs_hinted_audit(
        dataset_root=args.dataset_root, blind_manifest_path=args.blind_manifest,
        hinted_manifest_path=args.hinted_manifest, output_dir=args.output_dir,
    )
    print(f"Audited {result['summary']['dimension_count']} Episode/model dimensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
