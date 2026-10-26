from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..evaluation.preflight import _verified_content
from ..evaluation.release import validate_released_view
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from .conditions import derive_routed_rows, validate_condition_contract
from .diagnostics import evaluate_linear_probe, unrelated_control_result
from .metrics import (
    faithfulness_diagonal_bootstrap,
    faithfulness_matrix,
    paired_stratified_bootstrap,
    summarize_rows,
)
from .prepare_probe import _array_hash


def _verified_rows(
    path: Path, expected_ids: list[str], condition_id: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids or len(observed) != len(set(observed)):
        raise ValueError(f"report condition coverage differs: {condition_id}")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content) or row.get("condition_id") != condition_id:
            raise ValueError(f"report condition row changed: {condition_id} {row.get('record_id')}")
    return rows


def _artifact_path(source: dict[str, Any], label: str) -> Path:
    path = Path(str(source.get("path", "")))
    if not path.is_file() or sha256_file(path) != source.get("sha256"):
        raise ValueError(f"{label} is missing or changed")
    return path


def _macro_family_delta(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], families: list[str],
) -> float:
    left = {str(row["record_id"]): row for row in left_rows}
    right = {str(row["record_id"]): row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows) or set(left) != set(right):
        raise ValueError("macro-family delta requires identical unique records")
    effects = []
    for family in families:
        ids = [record_id for record_id, row in left.items() if row["model_family"] == family]
        if not ids or any(right[record_id]["model_family"] != family for record_id in ids):
            raise ValueError(f"macro-family delta lacks family coverage: {family}")
        effects.append(float(np.mean([
            float(bool(left[record_id]["correct"])) - float(bool(right[record_id]["correct"]))
            for record_id in ids
        ])))
    return float(np.mean(effects))


def query_type_leave_one_out(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], families: list[str],
) -> dict[str, float]:
    query_types = sorted({str(row["query_type"]) for row in left_rows})
    right = {str(row["record_id"]): row for row in right_rows}
    result: dict[str, float] = {}
    for omitted in query_types:
        left_subset = [row for row in left_rows if str(row["query_type"]) != omitted]
        ids = {str(row["record_id"]) for row in left_subset}
        right_subset = [right[record_id] for record_id in [str(row["record_id"]) for row in left_subset]]
        if ids != {str(row["record_id"]) for row in right_subset}:
            raise ValueError("query-type leave-one-out pairing differs")
        result[omitted] = _macro_family_delta(left_subset, right_subset, families)
    return result


def common_parse_success_delta(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], families: list[str],
) -> dict[str, Any]:
    right = {str(row["record_id"]): row for row in right_rows}
    left_subset = [
        row for row in left_rows
        if bool(row["parse_ok"]) and bool(right[str(row["record_id"])]["parse_ok"])
    ]
    right_subset = [right[str(row["record_id"])] for row in left_subset]
    return {
        "record_count": len(left_subset),
        "macro_family_accuracy_delta": _macro_family_delta(left_subset, right_subset, families),
    }


def _load_probe(manifest: dict[str, Any]) -> dict[str, Any]:
    path = _artifact_path(manifest["artifacts"]["probe"], "probe weights")
    with np.load(path) as values:
        return {
            "weights": values["weights"],
            "feature_mean": values["feature_mean"],
            "feature_scale": values["feature_scale"],
            "intercept": float(values["intercept"][0]),
            "l2": float(values["l2"][0]),
        }


def _verified_activations(
    source: dict[str, Any], rows: list[dict[str, Any]], label: str,
) -> np.ndarray:
    path = _artifact_path(source, label)
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or values.shape[0] != len(rows) or values.dtype != np.float32:
        raise ValueError(f"{label} has a different shape or dtype")
    for index, row in enumerate(rows):
        if row.get("activation_sha256") != _array_hash(values[index]):
            raise ValueError(f"{label} row changed: {row.get('record_id')}")
    return np.asarray(values)


def _probe_results(
    probe_manifest: dict[str, Any],
    sealed_rows: dict[str, list[dict[str, Any]]],
    sealed_summaries: dict[str, Any],
    controller: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    families: list[str],
    wrong_mapping: dict[str, str],
    random_primary: str,
) -> dict[str, Any]:
    probe = _load_probe(probe_manifest)
    expected_ids = [str(row["record_id"]) for row in sealed_rows["base"]]
    labels = np.asarray([
        1 if str(evaluator[record_id]["answer"]).lower() == "yes" else 0
        for record_id in expected_ids
    ], dtype=np.int64)
    base = _verified_activations(
        sealed_summaries["base"]["artifacts"]["activations"], sealed_rows["base"],
        "sealed base activations",
    )
    family_values = {
        family: _verified_activations(
            sealed_summaries[f"family_direction__{family}"]["artifacts"]["activations"],
            sealed_rows[f"family_direction__{family}"], f"sealed {family} activations",
        )
        for family in families
    }
    random_values = _verified_activations(
        sealed_summaries[random_primary]["artifacts"]["activations"],
        sealed_rows[random_primary], "sealed primary random activations",
    )
    oracle_correct = np.stack([
        family_values[str(controller[record_id]["model_family"])][index]
        for index, record_id in enumerate(expected_ids)
    ])
    oracle_wrong = np.stack([
        family_values[wrong_mapping[str(controller[record_id]["model_family"])]][index]
        for index, record_id in enumerate(expected_ids)
    ])
    values = {
        "base": base,
        "oracle_correct": oracle_correct,
        "oracle_wrong": oracle_wrong,
        random_primary: random_values,
    }
    results = {
        condition: evaluate_linear_probe(probe, activations, labels)
        for condition, activations in values.items()
    }
    base_metrics = results["base"]
    for metrics in results.values():
        metrics["accuracy_delta_vs_base"] = metrics["accuracy"] - base_metrics["accuracy"]
        metrics["auroc_delta_vs_base"] = metrics["auroc"] - base_metrics["auroc"]
        metrics["gold_margin_delta_vs_base"] = (
            metrics["mean_gold_margin"] - base_metrics["mean_gold_margin"]
        )
    return {
        "selected_layer": int(probe_manifest["selected_layer"]),
        "selected_l2": float(probe_manifest["selected_l2"]),
        "conditions": results,
        "interpretation": "diagnostic_only_not_a_substitute_for_generation_accuracy",
    }


def _condition_rows_from_raw(
    raw_conditions: dict[str, Any], expected_ids: list[str], condition_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if set(raw_conditions) != set(condition_ids):
        raise ValueError("raw run condition coverage differs from the frozen contract")
    return {
        condition_id: _verified_rows(
            _artifact_path(raw_conditions[condition_id]["artifacts"]["rows"], condition_id),
            expected_ids, condition_id,
        )
        for condition_id in condition_ids
    }


def assemble_report(
    charter_path: Path,
    control_charter_path: Path,
    raw_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    control_charter = yaml.safe_load(control_charter_path.read_text(encoding="utf-8"))
    contract = validate_condition_contract(charter)
    raw = _verified_content(
        read_json(raw_manifest_path), "raw_run_content_sha256", "sealed raw run",
    )
    if (
        raw.get("type") != "lollanet_cma_sealed_evaluation_raw_run_v1"
        or raw.get("status") != "complete_raw_conditions"
        or raw.get("code") != runtime_code_identity(Path(__file__).parent)
        or raw.get("dependency_lock") != dependency_lock()
        or raw.get("environment") != runtime_environment()
    ):
        raise ValueError("sealed raw run is incomplete or belongs to a different runtime")
    intent_path = _artifact_path(raw["intent"], "sealed run intent")
    intent = _verified_content(
        read_json(intent_path), "intent_content_sha256", "sealed run intent",
    )
    if (
        intent["charter_sha256"] != sha256_file(charter_path)
        or intent["control_charter_sha256"] != sha256_file(control_charter_path)
        or intent.get("code") != runtime_code_identity(Path(__file__).parent)
        or intent.get("dependency_lock") != dependency_lock()
        or intent.get("environment") != runtime_environment()
    ):
        raise ValueError("sealed raw run used different charters")
    report_path = output_dir / "report.json"
    existing_report = None
    if report_path.is_file():
        existing_report = _verified_content(
            read_json(report_path), "report_content_sha256", "sealed report",
        )
        if (
            existing_report.get("raw_manifest_sha256") != sha256_file(raw_manifest_path)
            or existing_report.get("code") != runtime_code_identity(Path(__file__).parent)
        ):
            raise ValueError("existing sealed report belongs to different inputs or code")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_source = Path(str(intent["preflight"]["path"]))
    if not preflight_source.is_file() or sha256_file(preflight_source) != intent["preflight"]["sha256"]:
        raise ValueError("sealed evaluation preflight is missing from the frozen runtime location")
    preflight = _verified_content(
        read_json(preflight_source), "sealed_evaluation_preflight_content_sha256",
        "sealed evaluation preflight",
    )
    probe_manifest_path = Path(str(intent["probe_manifest"]["path"]))
    if (
        probe_manifest_path.resolve() != Path(preflight["probe"]["path"]).resolve()
        or sha256_file(probe_manifest_path) != intent["probe_manifest"]["sha256"]
    ):
        raise ValueError("probe manifest differs from the sealed run intent")
    probe_manifest = _verified_content(
        read_json(probe_manifest_path), "probe_content_sha256", "hidden-state probe",
    )
    release_manifest_path = Path(str(intent["evaluation_release"]["path"]))
    if (
        release_manifest_path.resolve() != Path(preflight["evaluation_release"]["path"]).resolve()
        or sha256_file(release_manifest_path) != intent["evaluation_release"]["sha256"]
    ):
        raise ValueError("evaluation release differs from the sealed run intent")
    release = _verified_content(
        read_json(release_manifest_path), "evaluation_release_content_sha256", "evaluation release",
    )
    families = list(charter["families"])
    if set(raw.get("splits", {})) != set(charter["data"]["views"]):
        raise ValueError("raw run split coverage differs from the frozen charter")
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    derived_artifacts: dict[str, Any] = {}
    split_controller: dict[str, dict[str, dict[str, Any]]] = {}
    split_evaluator: dict[str, dict[str, dict[str, Any]]] = {}
    for split, view_contract in charter["data"]["views"].items():
        view_dir = Path(view_contract["path"])
        validate_released_view(view_dir, split)
        if (
            split not in release["views"]
            or sha256_file(view_dir / "manifest.json") != release["views"][split]["manifest_sha256"]
            or raw["splits"][split]["view_manifest_sha256"]
            != release["views"][split]["manifest_sha256"]
        ):
            raise ValueError(f"report view differs from the released raw-run source: {split}")
        view_manifest = read_json(view_dir / "manifest.json")
        prompts = read_jsonl(Path(view_manifest["artifacts"]["prompts"]["path"]))
        controllers = read_jsonl(Path(view_manifest["artifacts"]["controller"]["path"]))
        evaluators = read_jsonl(Path(view_manifest["artifacts"]["evaluator"]["path"]))
        expected_ids = [str(row["record_id"]) for row in prompts]
        controller = {str(row["record_id"]): row for row in controllers}
        evaluator = {str(row["record_id"]): row for row in evaluators}
        split_controller[split] = controller
        split_evaluator[split] = evaluator
        raw_conditions = raw["splits"][split]["conditions"]
        model_rows = _condition_rows_from_raw(
            raw_conditions, expected_ids, contract["model_run_condition_ids"],
        )
        family_source = {
            condition_id: model_rows[condition_id]
            for condition_id in model_rows if condition_id.startswith("family_direction__")
        }
        all_rows = dict(model_rows)
        derived_artifacts[split] = {}
        for condition_id in contract["derived_condition_ids"]:
            rows = derive_routed_rows(
                condition_id, expected_ids, controller, family_source,
                charter["conditions"]["wrong_mapping"],
                charter["conditions"]["shuffled_mapping"],
            )
            path = output_dir / "derived" / split / f"{condition_id}.jsonl"
            if path.is_file():
                if read_jsonl(path) != rows:
                    raise ValueError(f"existing derived rows differ: {split} {condition_id}")
            else:
                atomic_write_jsonl(path, rows)
            all_rows[condition_id] = _verified_rows(path, expected_ids, condition_id)
            derived_artifacts[split][condition_id] = {
                "path": str(path.resolve()), "sha256": sha256_file(path),
            }
        split_rows[split] = all_rows
        split_summaries[split] = {
            condition_id: summarize_rows(
                rows, families, int(charter["metrics"]["ece_bins"]),
            )
            for condition_id, rows in all_rows.items()
        }
    sealed = split_rows["sealed_test"]
    comparisons = {}
    for left, right in charter["success_gates"]["registered_comparisons"]:
        key = f"{left}_vs_{right}"
        comparisons[key] = paired_stratified_bootstrap(
            sealed[left], sealed[right], families,
            int(charter["statistics"]["paired_bootstrap_samples"]),
            int(charter["statistics"]["bootstrap_seed"]),
            float(charter["statistics"]["confidence_level"]),
        )
        comparisons[key]["passed"] = comparisons[key]["lower"] > 0
    family_directions = {
        family: sealed[f"family_direction__{family}"] for family in families
    }
    faithfulness = faithfulness_matrix(sealed["base"], family_directions, families)
    faithfulness["bootstrap"] = faithfulness_diagonal_bootstrap(
        sealed["base"], family_directions, families,
        int(charter["statistics"]["paired_bootstrap_samples"]),
        int(charter["statistics"]["bootstrap_seed"])
        + int(charter["statistics"]["faithfulness_bootstrap_seed_offset"]),
        float(charter["statistics"]["confidence_level"]),
    )
    ood = {}
    for split in ("ood_anticommonsense", "ood_nonsense"):
        delta = (
            split_summaries[split]["oracle_correct"]["macro_family_accuracy"]
            - split_summaries[split]["base"]["macro_family_accuracy"]
        )
        ood[split] = {"macro_family_accuracy_delta": delta, "nonreversal_passed": delta >= 0}
    leave_one_out = query_type_leave_one_out(sealed["oracle_correct"], sealed["base"], families)
    parser_common = common_parse_success_delta(sealed["oracle_correct"], sealed["base"], families)
    parse_failure_increase = (
        split_summaries["sealed_test"]["oracle_correct"]["parse_failure_rate"]
        - split_summaries["sealed_test"]["base"]["parse_failure_rate"]
    )
    random_sensitivity = {
        condition: (
            split_summaries["sealed_test"]["oracle_correct"]["macro_family_accuracy"]
            - split_summaries["sealed_test"][condition]["macro_family_accuracy"]
        )
        for condition in charter["conditions"]["random_sensitivity"]
    }
    control_manifest_path = Path(
        charter["readiness_blockers"]["unrelated_control_dataset"]["artifact"],
    )
    control_manifest = read_json(control_manifest_path)
    if (
        sha256_file(control_manifest_path) != raw["unrelated_control"]["manifest_sha256"]
        or sha256_file(control_manifest_path)
        != charter["readiness_blockers"]["unrelated_control_dataset"]["artifact_sha256"]
    ):
        raise ValueError("raw unrelated-control bundle differs from the frozen charter")
    expected_control_conditions = list(control_charter["evaluation"]["conditions"])
    if set(raw["unrelated_control"]["conditions"]) != set(expected_control_conditions):
        raise ValueError("raw unrelated-control condition coverage differs")
    control_ids = [
        str(row["record_id"])
        for row in read_jsonl(Path(control_manifest["artifacts"]["prompts"]["path"]))
    ]
    control_rows = {
        condition: _verified_rows(
            _artifact_path(raw["unrelated_control"]["conditions"][condition]["artifact"], condition),
            control_ids, condition,
        )
        for condition in expected_control_conditions
    }
    control_accuracy = {
        condition: float(np.mean([bool(row["correct"]) for row in rows]))
        for condition, rows in control_rows.items()
    }
    control_result = unrelated_control_result(
        control_accuracy["base"],
        {condition: value for condition, value in control_accuracy.items() if condition != "base"},
        float(control_charter["evaluation"]["maximum_allowed_accuracy_loss"]),
    )
    probe = _probe_results(
        probe_manifest, sealed, raw["splits"]["sealed_test"]["conditions"],
        split_controller["sealed_test"], split_evaluator["sealed_test"], families,
        charter["conditions"]["wrong_mapping"], charter["conditions"]["random_primary"],
    )
    gates = {
        "registered_comparisons": all(value["passed"] for value in comparisons.values()),
        "minimum_significant_diagonal_families": (
            faithfulness["bootstrap"]["significant_positive_count"]
            >= int(charter["success_gates"]["minimum_significant_diagonal_families"])
        ),
        "ood_nonreversal": all(value["nonreversal_passed"] for value in ood.values()),
        "random_sensitivity": all(value > 0 for value in random_sensitivity.values()),
        "query_type_robustness": all(value > 0 for value in leave_one_out.values()),
        "common_parse_success": parser_common["macro_family_accuracy_delta"] > 0,
        "parse_failure_rate": (
            parse_failure_increase
            <= float(charter["success_gates"]["maximum_oracle_parse_failure_rate_increase"])
        ),
        "unrelated_control": bool(control_result["passed"]),
    }
    report = {
        "type": "lollanet_cma_sealed_evaluation_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "phase2_eligible" if all(gates.values()) else "stopped_by_preregistered_gate",
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "evaluation_release_sha256": sha256_file(release_manifest_path),
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "condition_summaries": split_summaries,
        "registered_comparisons": comparisons,
        "faithfulness": faithfulness,
        "ood_nonreversal": ood,
        "random_sensitivity_macro_accuracy_delta": random_sensitivity,
        "query_type_leave_one_out_macro_accuracy_delta": leave_one_out,
        "common_parse_success": parser_common,
        "oracle_parse_failure_rate_increase": parse_failure_increase,
        "unrelated_control": control_result,
        "probe": probe,
        "success_gates": gates,
        "all_success_gates_passed": all(gates.values()),
        "derived_artifacts": derived_artifacts,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(), "environment": runtime_environment(),
    }
    report["report_content_sha256"] = stable_hash(report)
    if existing_report is not None:
        existing_comparable = dict(existing_report)
        expected_comparable = dict(report)
        for value in (existing_comparable, expected_comparable):
            value.pop("created_at", None)
            value.pop("report_content_sha256", None)
        if existing_comparable != expected_comparable:
            raise ValueError("existing sealed report differs from recomputed verified results")
        return existing_report
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the sealed LollaNet-CMA report.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("sealed_eval_charter.yaml"))
    parser.add_argument("--control-charter", type=Path, default=Path(__file__).with_name("control_set_charter.yaml"))
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_report(args.charter, args.control_charter, args.raw_manifest, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
