from __future__ import annotations

from typing import Any


REQUIRED_CONDITIONS = {
    "no_intervention",
    "prompting",
    "static_caa_diffmean",
    "fixed_reft",
    "always_on_low_rank",
    "closed_loop_low_rank",
    "matched_random_closed_loop",
}
REQUIRED_PERTURBATIONS = {
    "matched_random_subspace",
    "reverse_target_axis",
    "non_target_axis",
}
REQUIRED_ENDPOINTS = {
    "behavioral_recovery_probability",
    "recovery_latency_tokens",
    "final_target_axis_effect",
    "intervention_rate",
    "cumulative_control_energy",
}


def validate_pilot_charter(charter: dict[str, Any]) -> None:
    issues: list[str] = []
    boundary = charter.get("claim_boundary", {})
    if boundary.get("internal_attractor_claim_allowed") is not False:
        issues.append("internal attractor claims must remain locked")
    required_attractor = {
        "control_withdrawal", "persistence_after_withdrawal",
        "autonomous_recovery_after_second_perturbation",
    }
    if not required_attractor.issubset(set(boundary.get("attractor_gate_requires", []))):
        issues.append("attractor gate is missing persistence or autonomous-recovery evidence")
    scope = charter.get("pilot_scope", {})
    if scope.get("evaluation_access") != "forbidden" or scope.get("independent_unit") != "case_id":
        issues.append("pilot must forbid evaluation access and use case_id as n")
    actuator = charter.get("controller", {}).get("actuator", {})
    if actuator.get("prompt_conditioned_low_rank_update") is True:
        issues.append("actuator type must be named, not encoded as a boolean")
    if actuator.get("type") != "prompt_conditioned_low_rank_update":
        issues.append("pilot requires a prompt-conditioned low-rank actuator")
    if actuator.get("static_global_direction") is not False:
        issues.append("pilot must not collapse to a static global direction")
    conditions = set(charter.get("conditions", []))
    if not REQUIRED_CONDITIONS.issubset(conditions):
        issues.append(f"missing comparison conditions: {sorted(REQUIRED_CONDITIONS - conditions)}")
    perturbations = set(charter.get("perturbations", {}).get("families", []))
    if not REQUIRED_PERTURBATIONS.issubset(perturbations):
        issues.append(f"missing perturbation families: {sorted(REQUIRED_PERTURBATIONS - perturbations)}")
    endpoints = set(charter.get("primary_pilot_endpoints", []))
    if not REQUIRED_ENDPOINTS.issubset(endpoints):
        issues.append(f"missing primary endpoints: {sorted(REQUIRED_ENDPOINTS - endpoints)}")
    gate = charter.get("success_gate", {})
    for field in (
        "positive_control_required_first", "closed_loop_beats_static_and_always_on",
        "closed_loop_beats_matched_random", "behavioral_and_internal_recovery_required",
        "no_material_quality_collapse", "no_material_output_concentration", "development_only",
    ):
        if gate.get(field) is not True:
            issues.append(f"success_gate.{field} must be true")
    if issues:
        raise ValueError("invalid closed-loop pilot charter:\n- " + "\n- ".join(issues))

