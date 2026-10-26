from __future__ import annotations

from pathlib import Path

import pytest

from research.phase1.lollanet_cma.qualification.common import normalized_local_checkpoint
from research.phase1.lollanet_cma.qualification.freeze import _verify_snapshot
from research.phase1.lollanet_cma.qualification.parser import parse_yes_no
from research.phase1.lollanet_cma.qualification.readiness import (
    _checkpoint_artifact_errors,
    _checkpoint_manifest_errors,
    _local_revision,
)
from research.phase1.lollanet_cma.qualification.selection import _paired_macro_interval, _validate_rows
from research.phase1.lollanet_cma.qualification.runner import _load_prefix, _validate_non_thinking_candidate
from research.phase1.lollanet_cma.qualification.resign_checkpoint import resign_checkpoint


def test_yes_no_parser_prefers_unique_final_marker_and_rejects_conflicts():
    assert parse_yes_no("Reasoning\nFinal answer: yes")["parsed_answer"] == "yes"
    assert parse_yes_no("Maybe yes.\nNo")["parsed_answer"] == "no"
    assert parse_yes_no("Final answer: yes\nFinal answer: no")["parse_ok"] is False
    assert parse_yes_no("No answer supplied")["parse_ok"] is False


def test_paired_macro_interval_preserves_record_pairing():
    left = [
        {"record_id": f"a-{i}", "model_family": family, "correct": True}
        for family in ("a", "b") for i in range(4)
    ]
    right = [{**row, "correct": index % 2 == 0} for index, row in enumerate(left)]
    interval = _paired_macro_interval(left, right, samples=500, seed=7)
    assert interval["estimate"] == 0.5
    assert interval["lower"] > 0


def test_local_checkpoint_is_normalized_relative_to_charter(tmp_path: Path):
    charter = tmp_path / "configuration" / "qualification_charter.yaml"
    charter.parent.mkdir()
    charter.write_text("version: test\n", encoding="utf-8")
    assert normalized_local_checkpoint(charter, "../models/candidate") == (tmp_path / "models" / "candidate").resolve()


def test_local_revision_accepts_huggingface_snapshot_commit(tmp_path: Path):
    revision = "a" * 40
    checkpoint = tmp_path / "snapshots" / revision
    checkpoint.mkdir(parents=True)
    assert _local_revision(checkpoint) == revision


def test_checkpoint_readiness_rejects_metadata_only_directory(tmp_path: Path):
    (tmp_path / "REVISION").write_text("a" * 40, encoding="utf-8")
    assert _checkpoint_artifact_errors(tmp_path) == [
        "checkpoint_config_missing",
        "checkpoint_tokenizer_config_missing",
        "checkpoint_weights_missing",
    ]


def test_snapshot_verification_checks_lfs_sha256(tmp_path: Path):
    from types import SimpleNamespace

    payload = b"frozen weights"
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(payload)
    siblings = [
        SimpleNamespace(rfilename="config.json", size=2, lfs=None, blob_id="a"),
        SimpleNamespace(rfilename="tokenizer_config.json", size=2, lfs=None, blob_id="b"),
        SimpleNamespace(
            rfilename="model.safetensors",
            size=len(payload),
            lfs={"sha256": __import__("hashlib").sha256(payload).hexdigest()},
            blob_id="c",
        ),
    ]
    rows = _verify_snapshot(tmp_path, siblings)
    assert {row["path"] for row in rows} == {"config.json", "tokenizer_config.json", "model.safetensors"}


def test_readiness_rejects_checkpoint_manifest_file_change(tmp_path: Path):
    from research.phase1.lollanet_cma.qualification.common import atomic_write_json, stable_hash

    charter = tmp_path / "charter.yaml"
    charter.write_text("version: test\n", encoding="utf-8")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"original")
    candidate = {"id": "candidate", "model_id": "org/model", "revision": "a" * 40}
    manifest = {
        "candidate_id": "candidate",
        "model_id": "org/model",
        "revision": "a" * 40,
        "checkpoint": str(tmp_path.resolve()),
        "charter_sha256": __import__("hashlib").sha256(charter.read_bytes()).hexdigest(),
        "files": [{"path": "model.safetensors", "size": 8, "sha256": __import__("hashlib").sha256(b"original").hexdigest()}],
        "file_count": 1,
        "total_bytes": 8,
    }
    manifest["checkpoint_content_sha256"] = stable_hash(manifest)
    atomic_write_json(tmp_path / "checkpoint_manifest.json", manifest)
    assert _checkpoint_manifest_errors(tmp_path, candidate, charter) == []
    weight.write_bytes(b"modified")
    assert _checkpoint_manifest_errors(tmp_path, candidate, charter) == ["checkpoint_manifest_file_mismatch"]


def test_selection_rejects_duplicate_prediction_records():
    charter = {
        "qualification": {
            "records": 2,
            "hook_smoke_records": 1,
            "macro_families": ["family-a"],
        }
    }
    manifest = {"candidate_id": "candidate", "record_count": 2, "hook_smoke_records": 1}
    row = {
        "record_id": "same",
        "model_family": "family-a",
        "query_type": "query",
        "gold_answer": "yes",
        "parsed_answer": "yes",
        "parse_ok": True,
        "correct": True,
        "latency_seconds": 1.0,
        "peak_gpu_memory_bytes": 1,
    }
    with pytest.raises(ValueError, match="duplicate records"):
        _validate_rows(manifest, [row, dict(row)], [{"record_id": "same", "peak_gpu_memory_bytes": 1}], charter)


def test_resume_requires_exact_ordered_prefix(tmp_path: Path):
    from research.phase1.lollanet_cma.qualification.common import atomic_write_jsonl

    path = tmp_path / "partial.jsonl"
    atomic_write_jsonl(path, [{"record_id": "b"}])
    with pytest.raises(ValueError, match="exact ordered prefix"):
        _load_prefix(path, ["a", "b"], "test")
    atomic_write_jsonl(path, [{"record_id": "a"}])
    assert _load_prefix(path, ["a", "b"], "test") == [{"record_id": "a"}]


def test_qualification_rejects_thinking_candidates():
    _validate_non_thinking_candidate({"model_id": "org/plain-instruct", "reasoning_mode": "disabled"})
    with pytest.raises(ValueError, match="explicitly disable"):
        _validate_non_thinking_candidate({"model_id": "org/plain-instruct"})
    with pytest.raises(ValueError, match="forbidden"):
        _validate_non_thinking_candidate({"model_id": "org/model-thinking", "reasoning_mode": "disabled"})


def test_checkpoint_resign_rehashes_files_without_changing_payload(tmp_path: Path):
    import yaml

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    tokenizer = checkpoint / "tokenizer_config.json"
    weights = checkpoint / "model.safetensors"
    config.write_text("{}", encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"weights")
    candidate = {
        "id": "candidate",
        "model_id": "org/model",
        "revision": "a" * 40,
        "local_checkpoint": str(checkpoint),
    }
    charter = tmp_path / "charter.yaml"
    charter.write_text(yaml.safe_dump({"candidates": [candidate]}), encoding="utf-8")
    files = [
        {"path": path.name, "size": path.stat().st_size, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
        for path in (config, weights, tokenizer)
    ]
    manifest = {
        "candidate_id": "candidate",
        "model_id": "org/model",
        "revision": "a" * 40,
        "checkpoint": str(checkpoint.resolve()),
        "charter_sha256": "old",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
    }
    from research.phase1.lollanet_cma.qualification.common import atomic_write_json, stable_hash

    manifest["checkpoint_content_sha256"] = stable_hash(manifest)
    atomic_write_json(checkpoint / "checkpoint_manifest.json", manifest)
    before = weights.read_bytes()
    resigned = resign_checkpoint(charter, "candidate")
    assert resigned["charter_sha256"] == __import__("hashlib").sha256(charter.read_bytes()).hexdigest()
    assert resigned["resigned_from_checkpoint_content_sha256"] == manifest["checkpoint_content_sha256"]
    assert weights.read_bytes() == before
