from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research.phase1.lollanet_cma.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from research.phase1.lollanet_cma.evaluation.preflight import audit_sealed_release_preflight
from research.phase1.lollanet_cma.evaluation.release import (
    _materialize_split,
    release_evaluation_views,
    validate_released_view,
)


def _source_bundle(tmp_path: Path) -> tuple[Path, dict]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    splits = ("development", "sealed_test", "ood_anticommonsense", "ood_nonsense")
    prompts = [
        {"record_id": split, "split": split, "prompt": split, "source_variant": "main"}
        for split in splits
    ]
    controllers = [
        {"record_id": split, "model_family": "family", "query_type": "query"} for split in splits
    ]
    evaluators = [
        {"record_id": split, "answer": "yes", "model_id": split} for split in splits
    ]
    for name, rows in (("prompts", prompts), ("controller", controllers), ("evaluator", evaluators)):
        atomic_write_jsonl(bundle / f"{name}.jsonl", rows)
    manifest = {
        "type": "lollanet_cma_stage0_bundle_v1",
        "artifacts": {
            name: {"path": str(bundle / f"{name}.jsonl"), "sha256": sha256_file(bundle / f"{name}.jsonl")}
            for name in ("prompts", "controller", "evaluator")
        },
    }
    manifest["bundle_content_sha256"] = stable_hash(manifest)
    atomic_write_json(bundle / "manifest.json", manifest)
    return bundle, manifest


def _ready_authorization(tmp_path: Path, bundle: Path, source_manifest: dict) -> tuple[Path, dict]:
    development_path = tmp_path / "development.json"
    development_path.write_text("{}\n", encoding="utf-8")
    preflight = {
        "type": "lollanet_cma_sealed_release_preflight_v1",
        "sealed_release_gate_status": "ready",
        "unseal_authorized": True,
        "errors": [],
        "stage0": {
            "path": str(bundle.resolve()),
            "manifest_sha256": sha256_file(bundle / "manifest.json"),
            "bundle_content_sha256": source_manifest["bundle_content_sha256"],
        },
        "development": {
            "manifest_path": str(development_path.resolve()),
            "manifest_sha256": sha256_file(development_path),
        },
        "code": {},
        "dependency_lock": {},
        "environment": {},
    }
    preflight["sealed_release_preflight_content_sha256"] = stable_hash(preflight)
    preflight_path = tmp_path / "preflight.json"
    atomic_write_json(preflight_path, preflight)
    return preflight_path, preflight


def test_sealed_release_preflight_stays_blocked_without_development(tmp_path: Path):
    bundle, manifest = _source_bundle(tmp_path)
    charter = {
        "source_stage0": {
            "path": str(bundle),
            "manifest_sha256": sha256_file(bundle / "manifest.json"),
            "bundle_content_sha256": manifest["bundle_content_sha256"],
        },
        "prerequisites": {
            "development_manifest_type": "lollanet_cma_development_selection_v1",
            "development_status": "frozen",
        },
        "release": {"splits": {}},
    }
    charter_path = tmp_path / "release.yaml"
    charter_path.write_text(yaml.safe_dump(charter), encoding="utf-8")
    result = audit_sealed_release_preflight(
        charter_path,
        bundle,
        tmp_path / "missing-development.json",
        tmp_path / "preflight.json",
    )
    assert result["sealed_release_gate_status"] == "blocked_missing_development"
    assert result["unseal_authorized"] is False
    assert result["stage0"]["validation"] is None


def test_release_refuses_a_blocked_preflight(tmp_path: Path):
    bundle, _ = _source_bundle(tmp_path)
    charter_path = tmp_path / "release.yaml"
    charter_path.write_text(yaml.safe_dump({"release": {"splits": {}}}), encoding="utf-8")
    preflight = {
        "type": "lollanet_cma_sealed_release_preflight_v1",
        "sealed_release_gate_status": "blocked_missing_development",
        "unseal_authorized": False,
        "errors": ["development_manifest_missing"],
    }
    preflight["sealed_release_preflight_content_sha256"] = stable_hash(preflight)
    preflight_path = tmp_path / "preflight.json"
    atomic_write_json(preflight_path, preflight)
    with pytest.raises(ValueError, match="not ready"):
        release_evaluation_views(charter_path, bundle, preflight_path, tmp_path / "release.json")


def test_materialized_view_contains_only_one_authorized_split(tmp_path: Path):
    bundle, source_manifest = _source_bundle(tmp_path)
    preflight_path, preflight = _ready_authorization(tmp_path, bundle, source_manifest)
    output = tmp_path / "sealed-view"
    _materialize_split(
        "sealed_test",
        {"expected_records": 1, "expected_records_per_family": 1},
        ["family"],
        source_manifest,
        preflight_path,
        preflight,
        output,
    )
    assert validate_released_view(output, "sealed_test")["record_count"] == 1
    assert [row["record_id"] for row in read_jsonl(output / "prompts.jsonl")] == ["sealed_test"]
    assert [row["record_id"] for row in read_jsonl(output / "evaluator.jsonl")] == ["sealed_test"]


def test_released_view_detects_label_tampering(tmp_path: Path):
    bundle, source_manifest = _source_bundle(tmp_path)
    preflight_path, preflight = _ready_authorization(tmp_path, bundle, source_manifest)
    output = tmp_path / "ood-view"
    _materialize_split(
        "ood_nonsense",
        {"expected_records": 1, "expected_records_per_family": 1},
        ["family"],
        source_manifest,
        preflight_path,
        preflight,
        output,
    )
    (output / "evaluator.jsonl").write_text(
        '{"answer":"no","record_id":"ood_nonsense"}\n', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or changed"):
        validate_released_view(output, "ood_nonsense")


def test_formal_release_charter_freezes_all_three_evaluation_splits():
    charter = yaml.safe_load(Path(
        "research/phase1/lollanet_cma/evaluation/sealed_release_charter.yaml",
    ).read_text(encoding="utf-8"))
    assert set(charter["release"]["splits"]) == {
        "sealed_test", "ood_anticommonsense", "ood_nonsense",
    }
    assert charter["prerequisites"]["require_direction_specificity"] is True
    assert charter["release"]["evaluator_process_must_receive_only_released_views"] is True
