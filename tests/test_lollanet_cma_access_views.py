from __future__ import annotations

from pathlib import Path

import pytest

from research.phase1.lollanet_cma.access_views.views import create_access_view, validate_access_view
from research.phase1.lollanet_cma.common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file, stable_hash


def _source_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    prompts = [
        {"record_id": "q", "split": "qualification", "prompt": "q", "source_variant": "main"},
        {"record_id": "v", "split": "vector_train", "prompt": "v", "source_variant": "main"},
        {"record_id": "s", "split": "sealed_test", "prompt": "s", "source_variant": "main"},
    ]
    controllers = [
        {"record_id": row["record_id"], "model_family": "family", "query_type": "query"}
        for row in prompts
    ]
    evaluators = [
        {"record_id": row["record_id"], "answer": "yes", "model_id": row["record_id"]}
        for row in prompts
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
    return bundle


def test_access_view_materializes_only_the_requested_split(tmp_path: Path):
    source = _source_bundle(tmp_path)
    output = tmp_path / "qualification"
    create_access_view(source, "qualification", output)
    assert validate_access_view(output, "qualification")["record_count"] == 1
    assert [row["record_id"] for row in read_jsonl(output / "prompts.jsonl")] == ["q"]
    assert [row["record_id"] for row in read_jsonl(output / "evaluator.jsonl")] == ["q"]


def test_access_view_refuses_sealed_materialization_before_freeze(tmp_path: Path):
    with pytest.raises(ValueError, match="not allowed"):
        create_access_view(_source_bundle(tmp_path), "sealed_test", tmp_path / "sealed")


def test_access_view_detects_post_materialization_label_change(tmp_path: Path):
    source = _source_bundle(tmp_path)
    output = tmp_path / "qualification"
    create_access_view(source, "qualification", output)
    (output / "evaluator.jsonl").write_text('{"record_id":"q","answer":"no"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_access_view(output, "qualification")

