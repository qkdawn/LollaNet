from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any


SCORE_FIELDS = ("formulaic_0_4", "specificity_0_4", "usefulness_0_4", "validity_0_4")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_score_sheet(path: Path, mapping: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    aliases = [str(row.get("item_id", "")) for row in rows]
    if len(aliases) != len(set(aliases)) or set(aliases) != set(mapping):
        raise ValueError(f"score sheet coverage differs from its blind mapping: {path}")
    missing = []
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        alias = str(row["item_id"])
        required = [*SCORE_FIELDS, "exclude_0_1"]
        if any(str(row.get(field, "")).strip() == "" for field in required):
            missing.append(alias)
            continue
        scores = {field: float(row[field]) for field in SCORE_FIELDS}
        if any(value < 0 or value > 4 or value * 2 != round(value * 2) for value in scores.values()):
            raise ValueError(f"scores must be 0-4 in half-point increments: {alias}")
        exclude = int(row["exclude_0_1"])
        if exclude not in {0, 1}:
            raise ValueError(f"exclude must be 0 or 1: {alias}")
        parsed[mapping[alias]] = {
            **scores, "exclude_0_1": exclude, "notes": str(row.get("notes", "")).strip(),
        }
    return parsed, missing


def _tiebreak_alias(record_id: str) -> str:
    return "tie-" + sha256(f"template-attractor-tie:{record_id}".encode("utf-8")).hexdigest()[:16]


def _write_tiebreak(
    disagreements: list[str], trajectories: dict[str, dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    order = list(disagreements)
    random.Random(2026072599).shuffle(order)
    packet = []
    mapping = {}
    for record_id in order:
        alias = _tiebreak_alias(record_id)
        mapping[alias] = record_id
        packet.append({
            "item_id": alias,
            "prompt": trajectories[record_id]["prompt"],
            "response": trajectories[record_id]["response"],
        })
    packet_path = output_dir / "adjudicator_packet.jsonl"
    packet_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet), encoding="utf-8")
    scores_path = output_dir / "adjudicator_scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", *SCORE_FIELDS, "exclude_0_1", "notes"])
        writer.writerows([[row["item_id"], "", "", "", "", "", ""] for row in packet])
    _write_json(output_dir / "adjudicator_private_mapping.json", mapping)
    return {"packet": str(packet_path.resolve()), "scores": str(scores_path.resolve()), "count": len(packet)}


def _class_counts(labels: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "development", "evaluation"):
        rows = [row for row in labels if row["split"] == split and not row["exclude"]]
        counts[split] = {
            "negative_le_1_5": sum(row["formulaic"] <= 1.5 for row in rows),
            "ambiguous_2": sum(1.5 < row["formulaic"] < 2.5 for row in rows),
            "positive_ge_2_5": sum(row["formulaic"] >= 2.5 for row in rows),
        }
    return counts


def prepare_model_assisted_labels(
    annotation_dir: Path,
    trajectories_path: Path,
    scores_path: Path,
    output_dir: Path,
    model_id: str,
    packet: str = "annotator_a",
) -> dict[str, Any]:
    """Create discovery-only labels from one blinded model-annotation pass."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = json.loads((annotation_dir / "private_mapping.json").read_text(encoding="utf-8"))
    if packet not in mappings:
        raise ValueError(f"unknown blind packet mapping: {packet}")
    scores, missing = _read_score_sheet(scores_path, mappings[packet])
    if missing:
        manifest = {
            "type": "template_attractor_model_assisted_labels_v1",
            "status": "awaiting_complete_model_annotations",
            "missing_score_count": len(missing),
        }
        _write_json(output_dir / "label_manifest.json", manifest)
        return manifest

    source_rows = _read_jsonl(trajectories_path)
    trajectories = {str(row["record_id"]): row for row in source_rows}
    if set(scores) != set(trajectories):
        raise ValueError("model score sheet differs from trajectory coverage")
    labels = []
    for row in source_rows:
        record_id = str(row["record_id"])
        score = scores[record_id]
        labels.append({
            "record_id": record_id,
            "case_id": row["case_id"],
            "split": row["split"],
            **{field.removesuffix("_0_4"): score[field] for field in SCORE_FIELDS},
            "exclude": score["exclude_0_1"],
        })
    labels_path = output_dir / "model_assisted_labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8",
    )
    counts = _class_counts(labels)
    discovery_ready = all(
        counts[split]["negative_le_1_5"] >= 1 and counts[split]["positive_ge_2_5"] >= 1
        for split in ("train", "development")
    )
    manifest = {
        "type": "template_attractor_model_assisted_labels_v1",
        "status": "ready_for_discovery_probe_fit" if discovery_ready else "blocked_insufficient_class_coverage",
        "annotation_provenance": {
            "annotator_type": "language_model",
            "model_id": model_id,
            "passes": 1,
            "blind_packet": packet,
            "independent_human_annotation": False,
            "scores_path": str(scores_path.resolve()),
            "scores_sha256": sha256(scores_path.read_bytes()).hexdigest(),
        },
        "evidence_boundary": (
            "Discovery-only model-assisted labels. They do not establish human construct validity, "
            "inter-rater agreement, or confirmatory evaluation evidence."
        ),
        "class_definition": {
            "negative": "formulaic <= 1.5",
            "positive": "formulaic >= 2.5",
            "excluded_from_binary_probe": "1.5 < formulaic < 2.5",
        },
        "class_counts": counts,
        "required_discovery_splits": ["train", "development"],
        "labels": str(labels_path.resolve()),
    }
    _write_json(output_dir / "label_manifest.json", manifest)
    return manifest


def prepare_consensus_labels(annotation_dir: Path, trajectories_path: Path, output_dir: Path) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((annotation_dir / "private_mapping.json").read_text(encoding="utf-8"))
    scores = {}
    missing_counts = {}
    for annotator in ("annotator_a", "annotator_b"):
        values, missing = _read_score_sheet(annotation_dir / f"{annotator}_scores.csv", mapping[annotator])
        scores[annotator] = values
        missing_counts[annotator] = len(missing)
    if any(missing_counts.values()):
        manifest = {
            "type": "template_attractor_consensus_labels_v1",
            "status": "awaiting_complete_independent_annotations",
            "missing_score_counts": missing_counts,
        }
        _write_json(output_dir / "label_manifest.json", manifest)
        return manifest

    source_rows = _read_jsonl(trajectories_path)
    trajectories = {str(row["record_id"]): row for row in source_rows}
    expected = set(trajectories)
    if set(scores["annotator_a"]) != expected or set(scores["annotator_b"]) != expected:
        raise ValueError("completed score sheets differ from trajectory coverage")

    ordered_ids = [str(row["record_id"]) for row in source_rows]
    formulaic_a = [scores["annotator_a"][record_id]["formulaic_0_4"] for record_id in ordered_ids]
    formulaic_b = [scores["annotator_b"][record_id]["formulaic_0_4"] for record_id in ordered_ids]
    disagreements = [
        record_id for record_id, left, right in zip(ordered_ids, formulaic_a, formulaic_b)
        if abs(left - right) > 1.0
        or scores["annotator_a"][record_id]["exclude_0_1"] != scores["annotator_b"][record_id]["exclude_0_1"]
    ]
    agreement = {
        "quadratic_weighted_kappa": float(cohen_kappa_score(formulaic_a, formulaic_b, weights="quadratic")),
        "within_one_point_rate": float(sum(abs(a - b) <= 1 for a, b in zip(formulaic_a, formulaic_b)) / len(ordered_ids)),
        "adjudication_count": len(disagreements),
    }
    if disagreements:
        tiebreak = _write_tiebreak(disagreements, trajectories, output_dir)
        manifest = {
            "type": "template_attractor_consensus_labels_v1",
            "status": "awaiting_independent_adjudication",
            "agreement": agreement,
            "tiebreak": tiebreak,
        }
        _write_json(output_dir / "label_manifest.json", manifest)
        return manifest

    labels = []
    for record_id in ordered_ids:
        left = scores["annotator_a"][record_id]
        right = scores["annotator_b"][record_id]
        labels.append({
            "record_id": record_id,
            "case_id": trajectories[record_id]["case_id"],
            "split": trajectories[record_id]["split"],
            **{field.removesuffix("_0_4"): (left[field] + right[field]) / 2 for field in SCORE_FIELDS},
            "exclude": max(left["exclude_0_1"], right["exclude_0_1"]),
        })
    labels_path = output_dir / "consensus_labels.jsonl"
    labels_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8")
    class_counts = _class_counts(labels)
    fit_ready = all(
        counts["negative_le_1_5"] >= 2 and counts["positive_ge_2_5"] >= 2
        for counts in class_counts.values()
    )
    manifest = {
        "type": "template_attractor_consensus_labels_v1",
        "status": "ready_for_probe_fit" if fit_ready else "blocked_insufficient_class_coverage",
        "agreement": agreement,
        "class_definition": {"negative": "formulaic <= 1.5", "positive": "formulaic >= 2.5", "excluded_from_binary_probe": "1.5 < formulaic < 2.5"},
        "class_counts": class_counts,
        "labels": str(labels_path.resolve()),
    }
    _write_json(output_dir / "label_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate blind scores and prepare consensus labels.")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-scores", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--packet", default="annotator_a")
    args = parser.parse_args()
    if args.model_scores:
        if not args.model_id:
            parser.error("--model-id is required with --model-scores")
        result = prepare_model_assisted_labels(
            args.annotation, args.trajectories, args.model_scores, args.output, args.model_id, args.packet,
        )
    else:
        result = prepare_consensus_labels(args.annotation, args.trajectories, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
