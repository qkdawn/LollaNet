from __future__ import annotations

import csv
import json

import numpy as np

from research.phase1.template_attractor_pilot.annotation import build_annotation_delivery
from research.phase1.template_attractor_pilot.discovery_subset import (
    prepare_discovery_subset, seed_prior_model_scores,
)
from research.phase1.template_attractor_pilot.natural_trajectories import _sha256


def _score(path, value: str = "1") -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        row.update({
            "formulaic_0_4": value, "specificity_0_4": "3", "usefulness_0_4": "3",
            "validity_0_4": "3", "exclude_0_1": "0",
        })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_seed_prior_scores_reuses_exact_records_and_keeps_new_items_blind(tmp_path) -> None:
    old_records = [
        {"record_id": "r-0", "prompt": "p0", "response": "a0"},
        {"record_id": "r-1", "prompt": "p1", "response": "a1"},
    ]
    new_records = [*old_records, {"record_id": "r-2", "prompt": "p2", "response": "a2"}]
    prior = tmp_path / "prior"
    target = tmp_path / "target"
    build_annotation_delivery(old_records, prior)
    build_annotation_delivery(new_records, target)
    _score(prior / "annotator_a_scores.csv")
    output = tmp_path / "codex_scores.csv"
    result = seed_prior_model_scores(
        target, prior, prior / "annotator_a_scores.csv", output,
    )
    assert result["seeded_record_count"] == 2
    assert result["remaining_record_count"] == 1
    remaining = [
        json.loads(line) for line in open(result["remaining_packet"], encoding="utf-8")
    ]
    assert len(remaining) == 1
    assert set(remaining[0]) == {"item_id", "prompt", "response"}
    with output.open(encoding="utf-8-sig", newline="") as handle:
        scores = list(csv.DictReader(handle))
    assert sum(bool(row["formulaic_0_4"]) for row in scores) == 2


def test_prepare_discovery_subset_removes_evaluation_from_every_artifact(tmp_path) -> None:
    splits = ["train", "train", "development", "development", "evaluation", "evaluation"]
    records = [
        {
            "record_id": f"r-{index}", "case_id": f"c-{index // 2}", "split": split,
            "prompt": f"p{index}", "response": f"a{index}",
        }
        for index, split in enumerate(splits)
    ]
    trajectories = tmp_path / "trajectories.jsonl"
    prefixes = tmp_path / "prefixes.jsonl"
    activations = tmp_path / "activations.npz"
    activation_index = tmp_path / "activation-index.json"
    trajectories.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
    )
    prefixes.write_text(
        "".join(json.dumps({"record_id": row["record_id"], "early_4": "x"}) + "\n" for row in records),
        encoding="utf-8",
    )
    np.savez_compressed(activations, early_4__layer_0=np.arange(12).reshape(6, 2))
    activation_index.write_text(json.dumps({
        "record_ids": [row["record_id"] for row in records], "windows": ["early_4"], "layers": [0],
    }), encoding="utf-8")
    artifacts = {
        "trajectories": {"path": str(trajectories), "sha256": _sha256(trajectories)},
        "prefixes": {"path": str(prefixes), "sha256": _sha256(prefixes)},
        "activations": {"path": str(activations), "sha256": _sha256(activations)},
        "activation_index": {"path": str(activation_index), "sha256": _sha256(activation_index)},
    }
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(json.dumps({
        "status": "awaiting_independent_blind_annotations",
        "checkpoint": "checkpoint", "generation": {"samples_per_case": 2},
        "candidate_layers": [0], "early_windows": ["early_4"],
        "annotation_release_gate": {"passed": True, "observed_truncation_rate": 0.0},
        "artifacts": artifacts,
    }), encoding="utf-8")
    output = tmp_path / "subset"
    result = prepare_discovery_subset(run_manifest, output)
    assert not result["evaluation_split_present"]
    subset_rows = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
    assert len(subset_rows) == 4
    assert {row["split"] for row in subset_rows} == {"train", "development"}
    with np.load(output / "early_activations.npz") as archive:
        assert archive["early_4__layer_0"].shape == (4, 2)
    packets = [
        json.loads(line) for line in (output / "annotation" / "annotator_a_packet.jsonl").read_text().splitlines()
    ]
    assert len(packets) == 4
    assert all(set(row) == {"item_id", "prompt", "response"} for row in packets)
