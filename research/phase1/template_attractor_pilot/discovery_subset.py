from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .annotation import build_annotation_delivery
from .natural_trajectories import _atomic_json, _atomic_jsonl, _read_jsonl, _sha256


DISCOVERY_SPLITS = ("train", "development")
SCORE_COLUMNS = (
    "formulaic_0_4", "specificity_0_4", "usefulness_0_4", "validity_0_4", "exclude_0_1",
)


def prepare_discovery_subset(run_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("status") != "awaiting_independent_blind_annotations"
        or not source_manifest.get("annotation_release_gate", {}).get("passed")
    ):
        raise ValueError("source run has not passed annotation release")
    artifacts = source_manifest["artifacts"]
    for name in ("trajectories", "activations", "activation_index", "prefixes"):
        path = Path(str(artifacts[name]["path"]))
        if not path.is_file() or _sha256(path) != artifacts[name]["sha256"]:
            raise ValueError(f"source run artifact changed: {name}")

    source_rows = _read_jsonl(Path(str(artifacts["trajectories"]["path"])))
    source_prefixes = _read_jsonl(Path(str(artifacts["prefixes"]["path"])))
    source_index = json.loads(Path(str(artifacts["activation_index"]["path"])).read_text(encoding="utf-8"))
    source_ids = [str(row["record_id"]) for row in source_rows]
    if source_ids != [str(row["record_id"]) for row in source_prefixes]:
        raise ValueError("source trajectory and prefix order differ")
    if source_ids != [str(value) for value in source_index["record_ids"]]:
        raise ValueError("source trajectory and activation order differ")

    selected_indices = np.asarray([
        index for index, row in enumerate(source_rows) if str(row["split"]) in DISCOVERY_SPLITS
    ], dtype=np.int64)
    selected_rows = [source_rows[int(index)] for index in selected_indices]
    selected_prefixes = [source_prefixes[int(index)] for index in selected_indices]
    if not selected_rows or any(row["split"] == "evaluation" for row in selected_rows):
        raise ValueError("discovery subset selection is empty or contains evaluation records")

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "trajectories.jsonl"
    prefixes_path = output_dir / "early_text_prefixes.jsonl"
    activation_path = output_dir / "early_activations.npz"
    activation_index_path = output_dir / "activation_index.json"
    _atomic_jsonl(trajectories_path, selected_rows)
    _atomic_jsonl(prefixes_path, selected_prefixes)
    with np.load(Path(str(artifacts["activations"]["path"]))) as archive:
        subset_arrays = {
            key: np.asarray(archive[key][selected_indices], dtype=np.float32)
            for key in archive.files
        }
    np.savez_compressed(activation_path, **subset_arrays)
    _atomic_json(activation_index_path, {
        "type": "template_attractor_discovery_activation_index_v1",
        "record_ids": [str(row["record_id"]) for row in selected_rows],
        "windows": source_index["windows"],
        "layers": source_index["layers"],
        "activation_sha256": _sha256(activation_path),
        "evaluation_records": 0,
    })

    annotation_manifest = build_annotation_delivery(selected_rows, output_dir / "annotation")
    audit = {
        "type": "template_attractor_discovery_subset_audit_v1",
        "record_count": len(selected_rows),
        "case_count": len({str(row["case_id"]) for row in selected_rows}),
        "split_counts": {
            split: sum(str(row["split"]) == split for row in selected_rows)
            for split in DISCOVERY_SPLITS
        },
        "evaluation_records": 0,
        "samples_per_case": source_manifest["generation"]["samples_per_case"],
    }
    audit_path = output_dir / "subset_audit.json"
    _atomic_json(audit_path, audit)
    subset_artifacts = {
        "trajectories": {"path": str(trajectories_path.resolve()), "sha256": _sha256(trajectories_path)},
        "activations": {"path": str(activation_path.resolve()), "sha256": _sha256(activation_path)},
        "activation_index": {"path": str(activation_index_path.resolve()), "sha256": _sha256(activation_index_path)},
        "prefixes": {"path": str(prefixes_path.resolve()), "sha256": _sha256(prefixes_path)},
        "audit": {"path": str(audit_path.resolve()), "sha256": _sha256(audit_path)},
        "annotation_manifest": annotation_manifest,
    }
    manifest = {
        "type": "template_attractor_natural_discovery_subset_v1",
        "status": "awaiting_independent_blind_annotations",
        "checkpoint": source_manifest["checkpoint"],
        "generation": source_manifest["generation"],
        "candidate_layers": source_manifest["candidate_layers"],
        "early_windows": source_manifest["early_windows"],
        "included_splits": list(DISCOVERY_SPLITS),
        "evaluation_split_present": False,
        "source_manifest": {
            "path": str(run_manifest_path.resolve()), "sha256": _sha256(run_manifest_path),
        },
        "artifacts": subset_artifacts,
        "annotation_release_gate": {
            "passed": True,
            "inherited_from_source": True,
            "source_observed_truncation_rate": source_manifest["annotation_release_gate"][
                "observed_truncation_rate"
            ],
        },
        "boundary": (
            "This artifact contains train and development records only. It cannot release or score "
            "evaluation data."
        ),
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    return manifest


def seed_prior_model_scores(
    target_annotation_dir: Path,
    prior_annotation_dir: Path,
    prior_scores_path: Path,
    output_scores_path: Path,
    packet: str = "annotator_a",
) -> dict[str, Any]:
    target_mapping = json.loads(
        (target_annotation_dir / "private_mapping.json").read_text(encoding="utf-8")
    )[packet]
    prior_mapping = json.loads(
        (prior_annotation_dir / "private_mapping.json").read_text(encoding="utf-8")
    )[packet]
    with prior_scores_path.open(encoding="utf-8-sig", newline="") as handle:
        prior_rows = list(csv.DictReader(handle))
    prior_by_record = {}
    for row in prior_rows:
        alias = str(row["item_id"])
        if alias not in prior_mapping:
            raise ValueError("prior score alias is absent from its private mapping")
        if any(str(row.get(field, "")).strip() == "" for field in SCORE_COLUMNS):
            continue
        prior_by_record[str(prior_mapping[alias])] = row

    template_path = target_annotation_dir / f"{packet}_scores.csv"
    with template_path.open(encoding="utf-8-sig", newline="") as handle:
        target_rows = list(csv.DictReader(handle))
        fieldnames = list(target_rows[0])
    seeded = 0
    remaining_aliases = []
    for row in target_rows:
        alias = str(row["item_id"])
        record_id = str(target_mapping[alias])
        prior = prior_by_record.get(record_id)
        if prior is None:
            remaining_aliases.append(alias)
            continue
        for field in (*SCORE_COLUMNS, "notes"):
            row[field] = prior.get(field, "")
        seeded += 1
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with output_scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(target_rows)

    packet_path = target_annotation_dir / f"{packet}_packet.jsonl"
    packet_rows = _read_jsonl(packet_path)
    remaining = [row for row in packet_rows if str(row["item_id"]) in set(remaining_aliases)]
    remaining_packet_path = output_scores_path.with_name(output_scores_path.stem + "_remaining_packet.jsonl")
    _atomic_jsonl(remaining_packet_path, remaining)
    result = {
        "type": "template_attractor_seeded_model_score_sheet_v1",
        "status": "awaiting_incremental_blind_model_annotations" if remaining else "complete",
        "seeded_record_count": seeded,
        "remaining_record_count": len(remaining),
        "packet": packet,
        "remaining_packet": str(remaining_packet_path.resolve()),
        "output_scores": str(output_scores_path.resolve()),
        "prior_scores": {
            "path": str(prior_scores_path.resolve()),
            "sha256": sha256(prior_scores_path.read_bytes()).hexdigest(),
        },
        "blind_boundary": (
            "The incremental packet contains prompt, response, and anonymous item_id only; split and "
            "record provenance remain absent."
        ),
    }
    _atomic_json(output_scores_path.with_suffix(".manifest.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evaluation-free discovery artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subset = subparsers.add_parser("subset")
    subset.add_argument("--run-manifest", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    seed = subparsers.add_parser("seed-scores")
    seed.add_argument("--target-annotation", type=Path, required=True)
    seed.add_argument("--prior-annotation", type=Path, required=True)
    seed.add_argument("--prior-scores", type=Path, required=True)
    seed.add_argument("--output-scores", type=Path, required=True)
    seed.add_argument("--packet", default="annotator_a")
    args = parser.parse_args()
    if args.command == "subset":
        result = prepare_discovery_subset(args.run_manifest, args.output)
    else:
        result = seed_prior_model_scores(
            args.target_annotation, args.prior_annotation, args.prior_scores,
            args.output_scores, args.packet,
        )
    print(result["status"])


if __name__ == "__main__":
    main()
