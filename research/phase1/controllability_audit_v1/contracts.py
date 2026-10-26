from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml


AXES = ("specificity", "independence", "non_template")
REQUIRED_CONDITIONS = {
    "no_intervention",
    "prompting",
    "caa_diffmean",
    "reft",
    "sae_subspace",
    "matched_random_subspace",
}
RANDOM_MATCH_FIELDS = {"layer", "rank", "update_norm", "token_window", "injection_policy"}
OPEN_SPLITS = {"train", "development"}


class CharterValidationError(ValueError):
    """Raised when a charter could permit an invalid causal claim."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("invalid controllability-audit charter:\n- " + "\n- ".join(issues))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_charter(path: Path) -> dict[str, Any]:
    charter = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(charter, dict):
        raise CharterValidationError(["charter root must be a mapping"])
    return charter


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def validate_charter(charter: dict[str, Any]) -> None:
    issues: list[str] = []

    target_axes = _get(charter, "constructs", "target_axes", default=[])
    axis_names = tuple(row.get("name") for row in target_axes if isinstance(row, dict))
    if axis_names != AXES:
        issues.append(f"target axes must remain separate and ordered as {AXES}")
    if any(not row.get("separate_label") for row in target_axes if isinstance(row, dict)):
        issues.append("every target axis must have a separate label")
    if _get(charter, "constructs", "composite_quality_score") is not False:
        issues.append("constructs.composite_quality_score must be false")

    positive = _get(charter, "constructs", "positive_control", default={})
    if not positive.get("name") or positive.get("included_in_target_axes") is not False:
        issues.append("a positive control distinct from the target axes is required")

    conditions = _get(charter, "conditions", default=[])
    names = {row.get("name") for row in conditions if isinstance(row, dict) and row.get("required")}
    missing = REQUIRED_CONDITIONS - names
    if missing:
        issues.append(f"missing required intervention conditions: {sorted(missing)}")

    splits = _get(charter, "splits", default={})
    if splits.get("grouping_key") != "case_id" or splits.get("case_disjoint") is not True:
        issues.append("splits must be case-disjoint using case_id")
    if splits.get("selection_split") != "train" or splits.get("tuning_split") != "development":
        issues.append("method fitting/selection must use train and tuning must use development")
    evaluation = splits.get("evaluation", {})
    if evaluation.get("sealed") is not True or evaluation.get("accessible_during_development") is not False:
        issues.append("evaluation must remain sealed and inaccessible during development")
    if evaluation.get("release_mode") != "once_after_freeze":
        issues.append("evaluation release must be one-time and after protocol freeze")

    random_control = _get(charter, "controls", "matched_random_subspace", default={})
    missing_matches = RANDOM_MATCH_FIELDS - set(random_control.get("match", []))
    if missing_matches:
        issues.append(f"random subspace control is not fully matched: {sorted(missing_matches)}")
    if int(random_control.get("minimum_draws", 0)) < 1:
        issues.append("at least one matched random-subspace draw is required")
    if _get(charter, "controls", "positive_control_required_before_target_claims") is not True:
        issues.append("positive control must pass before target-axis causal claims")

    reporting = _get(charter, "reporting", default={})
    for field in ("target_effects_separate_by_axis", "cross_axis_effect_matrix", "quality_preservation_separate"):
        if reporting.get(field) is not True:
            issues.append(f"reporting.{field} must be true")
    if reporting.get("composite_overall_quality_forbidden") is not True:
        issues.append("a composite overall-quality result must be explicitly forbidden")

    statistics = _get(charter, "statistics", default={})
    if statistics.get("independent_unit") != "case_id":
        issues.append("the independent statistical unit must be case_id")
    repetitions = set(statistics.get("within_case_repetitions", []))
    if not {"seed", "response", "token"}.issubset(repetitions):
        issues.append("seed, response and token must be declared within-case repetitions")
    if not statistics.get("report_effect_sizes") or not statistics.get("report_confidence_intervals"):
        issues.append("effect sizes and confidence intervals must be reported")
    family = statistics.get("comparison_family", {})
    if not family.get("primary") or not family.get("multiplicity_strategy"):
        issues.append("the primary comparison family and multiplicity strategy must be declared")

    annotation = _get(charter, "annotation", default={})
    if annotation.get("model_assisted_labels_are_human_validation") is not False:
        issues.append("model-assisted labels cannot be treated as human validation")
    if annotation.get("human_validation_required") is not True:
        issues.append("human validation must be required")
    if int(annotation.get("independent_blind_annotators", 0)) < 2:
        issues.append("human validation requires at least two independent blind annotators")
    if annotation.get("report_agreement") is not True:
        issues.append("annotator agreement must be reported")

    boundary = _get(charter, "evidence_boundary", default={})
    if boundary.get("prior_versions_are_discovery_only") is not True:
        issues.append("prior exploratory versions must remain discovery-only")
    if boundary.get("formal_evaluation_accessed") is not False:
        issues.append("formal evaluation must not have been accessed")
    if boundary.get("formal_evaluation_results_available") is not False:
        issues.append("formal evaluation results must not be available before release")

    if _get(charter, "paper", "attractor_claim_allowed") is not False:
        issues.append("attractor claims are not allowed by this audit charter")

    if issues:
        raise CharterValidationError(issues)


def preflight_open_data(charter: dict[str, Any]) -> dict[str, Any]:
    """Inventory declared open JSONL sources without accessing any evaluation source."""
    validate_charter(charter)
    report: dict[str, Any] = {"study_id": charter.get("study_id"), "sources": {}}
    source_case_splits: dict[str, dict[str, str]] = {}
    for source_name, spec in _get(charter, "open_data", default={}).items():
        path = Path(spec["path"])
        source_report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            source_report["status"] = "missing"
            report["sources"][source_name] = source_report
            continue

        counts: Counter[str] = Counter()
        records = 0
        case_ids: set[str] = set()
        case_splits: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                split = str(row.get("split", "missing"))
                if split not in OPEN_SPLITS:
                    raise CharterValidationError(
                        [f"open source {source_name} contains forbidden split {split!r} at line {line_number}"]
                    )
                counts[split] += 1
                records += 1
                if row.get("case_id") is not None:
                    case_id = str(row["case_id"])
                    prior_split = case_splits.setdefault(case_id, split)
                    if prior_split != split:
                        raise CharterValidationError(
                            [f"open source {source_name} places case_id {case_id!r} in multiple splits"]
                        )
                    case_ids.add(case_id)

        expected = spec.get("expected_records")
        source_report.update(
            records=records,
            cases=len(case_ids),
            splits=dict(sorted(counts.items())),
            sha256=_sha256_file(path),
            expected_records=expected,
            status="ready" if expected is None or records == int(expected) else "count_mismatch",
        )
        report["sources"][source_name] = source_report
        source_case_splits[source_name] = case_splits

    populated = [(name, mapping) for name, mapping in source_case_splits.items() if mapping]
    if populated:
        reference_name, reference = populated[0]
        for source_name, mapping in populated[1:]:
            if mapping != reference:
                raise CharterValidationError(
                    [f"case/split mapping differs between open sources {reference_name} and {source_name}"]
                )
    return report
