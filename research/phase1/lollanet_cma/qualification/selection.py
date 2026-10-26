from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

import numpy as np
import yaml

from .common import atomic_write_json, read_json, read_jsonl, sha256_file, stable_hash


_SHARED_MANIFEST_FIELDS = (
    "charter",
    "stage0_manifest_sha256",
    "access_view_manifest_sha256",
    "readiness_sha256",
    "quantization",
    "compute_dtype",
    "batch_size",
    "maximum_input_tokens",
    "maximum_new_tokens",
    "sampling",
    "seed",
    "hook_smoke_records",
    "record_count",
    "code",
    "dependency_lock",
    "environment",
)


def _macro_accuracy(rows: list[dict[str, Any]]) -> float:
    families = sorted({row["model_family"] for row in rows})
    return float(np.mean([np.mean([item["correct"] for item in rows if item["model_family"] == family]) for family in families]))


def _paired_macro_interval(
    left: list[dict[str, Any]], right: list[dict[str, Any]], samples: int, seed: int,
) -> dict[str, float]:
    right_by_id = {row["record_id"]: row for row in right}
    if set(right_by_id) != {row["record_id"] for row in left}:
        raise ValueError("qualification candidates do not cover identical records")
    family_rows = {
        family: [row for row in left if row["model_family"] == family]
        for family in sorted({row["model_family"] for row in left})
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        left_sample: list[dict[str, Any]] = []
        right_sample: list[dict[str, Any]] = []
        for rows in family_rows.values():
            sampled = rng.integers(0, len(rows), len(rows))
            left_sample.extend(rows[position] for position in sampled)
            right_sample.extend(right_by_id[rows[position]["record_id"]] for position in sampled)
        values[index] = _macro_accuracy(left_sample) - _macro_accuracy(right_sample)
    return {
        "estimate": _macro_accuracy(left) - _macro_accuracy(right),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def _validate_rows(
    manifest: dict[str, Any], rows: list[dict[str, Any]], smoke: list[dict[str, Any]], charter: dict[str, Any],
) -> dict[str, tuple[str, str, str]]:
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"qualification predictions contain duplicate records: {manifest['candidate_id']}")
    expected_count = int(charter["qualification"]["records"])
    if len(rows) != expected_count or int(manifest["record_count"]) != expected_count:
        raise ValueError(f"qualification predictions are incomplete: {manifest['candidate_id']}")
    smoke_ids = [str(row["record_id"]) for row in smoke]
    expected_smoke_count = int(charter["qualification"]["hook_smoke_records"])
    if (
        len(smoke_ids) != expected_smoke_count
        or len(smoke_ids) != len(set(smoke_ids))
        or not set(smoke_ids).issubset(record_ids)
        or int(manifest["hook_smoke_records"]) != expected_smoke_count
    ):
        raise ValueError(f"qualification hook smoke is incomplete or duplicated: {manifest['candidate_id']}")
    expected_families = set(charter["qualification"]["macro_families"])
    if {row["model_family"] for row in rows} != expected_families:
        raise ValueError(f"qualification predictions have unexpected model-family coverage: {manifest['candidate_id']}")
    family_accuracy = {
        family: float(np.mean([bool(row["correct"]) for row in rows if row["model_family"] == family]))
        for family in sorted(expected_families)
    }
    for row in rows:
        expected_correct = bool(row["parse_ok"] and row.get("parsed_answer") == row["gold_answer"])
        if bool(row["correct"]) != expected_correct:
            raise ValueError(f"qualification correctness field is inconsistent: {manifest['candidate_id']}")
    if manifest["family_accuracy"] != family_accuracy or not np.isclose(
        float(manifest["macro_family_accuracy"]), float(np.mean(list(family_accuracy.values())))
    ):
        raise ValueError(f"qualification accuracy summary does not match predictions: {manifest['candidate_id']}")
    parse_failure_rate = float(np.mean([not bool(row["parse_ok"]) for row in rows]))
    median_latency = float(statistics.median(float(row["latency_seconds"]) for row in rows))
    peak_memory = max(int(row["peak_gpu_memory_bytes"]) for row in [*smoke, *rows])
    if (
        not np.isclose(float(manifest["parse_failure_rate"]), parse_failure_rate)
        or not np.isclose(float(manifest["median_latency_seconds"]), median_latency)
        or int(manifest["peak_gpu_memory_bytes"]) != peak_memory
    ):
        raise ValueError(f"qualification runtime summary does not match predictions: {manifest['candidate_id']}")
    return {
        str(row["record_id"]): (str(row["model_family"]), str(row["query_type"]), str(row["gold_answer"]))
        for row in rows
    }


def select_candidate(charter_path: Path, run_dirs: list[Path], output_path: Path) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    expected_ids = {row["id"] for row in charter["candidates"]}
    runs: list[tuple[dict[str, Any], list[dict[str, Any]], Path]] = []
    expected_by_id = {row["id"]: row for row in charter["candidates"]}
    shared_contract: dict[str, Any] | None = None
    shared_records: dict[str, tuple[str, str, str]] | None = None
    shared_smoke_ids: set[str] | None = None
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        manifest = read_json(manifest_path)
        content = dict(manifest)
        claimed = content.pop("run_content_sha256", None)
        if claimed != stable_hash(content) or not manifest.get("hard_gate_pass") or not all(manifest.get("hard_gates", {}).values()):
            raise ValueError(f"qualification run is invalid or ineligible: {manifest_path}")
        candidate_id = manifest.get("candidate_id")
        candidate = expected_by_id.get(candidate_id)
        if candidate is None or manifest.get("model_id") != candidate["model_id"] or manifest.get("revision") != candidate["revision"]:
            raise ValueError(f"qualification run does not match a preregistered candidate: {manifest_path}")
        if manifest.get("charter", {}).get("sha256") != sha256_file(charter_path):
            raise ValueError(f"qualification run used a different charter: {manifest_path}")
        contract = {field: manifest.get(field) for field in _SHARED_MANIFEST_FIELDS}
        if shared_contract is None:
            shared_contract = contract
        elif contract != shared_contract:
            raise ValueError(f"qualification runs do not share one frozen execution contract: {manifest_path}")
        predictions_path = Path(manifest["artifacts"]["predictions"]["path"])
        if not predictions_path.is_file() or sha256_file(predictions_path) != manifest["artifacts"]["predictions"]["sha256"]:
            raise ValueError(f"qualification predictions are missing or changed: {manifest_path}")
        smoke_path = Path(manifest["artifacts"]["hook_smoke"]["path"])
        if not smoke_path.is_file() or sha256_file(smoke_path) != manifest["artifacts"]["hook_smoke"]["sha256"]:
            raise ValueError(f"qualification hook smoke is missing or changed: {manifest_path}")
        intent_path = Path(manifest["artifacts"]["run_intent"]["path"])
        if not intent_path.is_file() or sha256_file(intent_path) != manifest["artifacts"]["run_intent"]["sha256"]:
            raise ValueError(f"qualification run intent is missing or changed: {manifest_path}")
        intent = read_json(intent_path)
        intent_content = dict(intent)
        intent_claimed = intent_content.pop("intent_content_sha256", None)
        if (
            intent_claimed != stable_hash(intent_content)
            or intent.get("candidate_id") != manifest.get("candidate_id")
            or intent.get("readiness_sha256") != manifest.get("readiness_sha256")
            or intent.get("access_view_manifest_sha256") != manifest.get("access_view_manifest_sha256")
            or intent.get("code") != manifest.get("code")
            or intent.get("environment") != manifest.get("environment")
        ):
            raise ValueError(f"qualification run intent does not match its manifest: {manifest_path}")
        access_view_path = Path(manifest["artifacts"]["access_view_manifest"]["path"])
        if (
            not access_view_path.is_file()
            or sha256_file(access_view_path) != manifest["artifacts"]["access_view_manifest"]["sha256"]
            or sha256_file(access_view_path) != manifest.get("access_view_manifest_sha256")
        ):
            raise ValueError(f"qualification access view is missing or changed: {manifest_path}")
        rows = read_jsonl(predictions_path)
        smoke = read_jsonl(smoke_path)
        records = _validate_rows(manifest, rows, smoke, charter)
        smoke_ids = {str(row["record_id"]) for row in smoke}
        if shared_records is None:
            shared_records = records
            shared_smoke_ids = smoke_ids
        elif records != shared_records or smoke_ids != shared_smoke_ids:
            raise ValueError("qualification candidates do not cover identical records and hook smoke")
        runs.append((manifest, rows, manifest_path))
    if {manifest["candidate_id"] for manifest, _, _ in runs} != expected_ids or len(runs) != len(expected_ids):
        raise ValueError("selection requires exactly one eligible run for every preregistered candidate")
    ranked = sorted(runs, key=lambda item: float(item[0]["macro_family_accuracy"]), reverse=True)
    top_manifest, top_rows, _ = ranked[0]
    intervals = {
        manifest["candidate_id"]: _paired_macro_interval(
            top_rows, rows, int(charter["selection"]["bootstrap_samples"]), int(charter["selection"]["seed"]),
        )
        for manifest, rows, _ in ranked[1:]
    }
    second_manifest, _, _ = ranked[1]
    interval = intervals[second_manifest["candidate_id"]]
    tie_candidates = [ranked[0], *[
        run for run in ranked[1:]
        if intervals[run[0]["candidate_id"]]["lower"] <= 0 <= intervals[run[0]["candidate_id"]]["upper"]
    ]]
    tie = len(tie_candidates) > 1
    if tie:
        tie_candidates.sort(key=lambda item: (
            int(item[0]["peak_gpu_memory_bytes"]),
            float(item[0]["median_latency_seconds"]),
            float(item[0]["parse_failure_rate"]),
        ))
    selected = tie_candidates[0][0]
    report = {
        "type": "lollanet_cma_base_model_qualification_v1",
        "status": "frozen_before_vector_training",
        "selected_candidate_id": selected["candidate_id"],
        "selected_model_id": selected["model_id"],
        "selected_revision": selected["revision"],
        "selection_reason": "preregistered_tie_break" if tie else "macro_family_accuracy_with_paired_ci",
        "top_vs_second_paired_bootstrap": interval,
        "paired_bootstrap_vs_top": intervals,
        "candidate_summaries": [
            {
                key: manifest[key] for key in (
                    "candidate_id", "model_id", "revision", "macro_family_accuracy", "family_accuracy",
                    "peak_gpu_memory_bytes", "median_latency_seconds", "parse_failure_rate",
                )
            }
            for manifest, _, _ in ranked
        ],
        "sources": [
            {"manifest_path": str(path), "manifest_sha256": sha256_file(path)} for _, _, path in runs
        ],
        "charter_sha256": sha256_file(charter_path),
    }
    report["qualification_content_sha256"] = stable_hash(report)
    atomic_write_json(output_path, report)
    return report
