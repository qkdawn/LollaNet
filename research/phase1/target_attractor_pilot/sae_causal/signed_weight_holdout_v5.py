from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers

from .contracts import read_jsonl
from .fit_signed_weights_v5 import weighted_activation_update
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom
from .signed_dose_train_v4 import DIRECTIONS
from .subspace_interchange_smoke_v4 import _generate, norm_match


AXIS = "non_template"
ALPHAS = (0.5, 1.0, 2.0)
METHODS = ("signed_weighted", "equal_weight", "random_weighted")


def _alpha_label(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def select_holdout_pairs(
    discovery: dict[str, Any], fit: dict[str, Any], count: int = 2
) -> list[dict[str, Any]]:
    if fit.get("development_accessed_during_fit") is not False:
        raise ValueError("fit accessed development")
    if not fit["axes"][AXIS]["fit"]["passes_train_only_gate"]:
        raise ValueError(f"{AXIS} did not pass the train-only fit gate")
    allowed = set(fit["axes"][AXIS]["internal_holdout_pair_ids"])
    pairs = [
        pair for pair in discovery["pairs"][AXIS]
        if pair["split"] == "train" and str(pair["pair_id"]) in allowed
    ]
    pairs = sorted(pairs, key=lambda pair: str(pair["pair_id"]))
    if len(pairs) < count:
        raise ValueError(f"only {len(pairs)} eligible internal holdout pairs")
    selected = pairs[:count]
    fit_cases = set(fit["axes"][AXIS].get("fit_case_ids", []))
    if fit_cases.intersection(str(pair["case_id"]) for pair in selected):
        raise ValueError("internal holdout overlaps fit cases")
    return selected


def build_plan(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for pair in pairs:
        for direction, source_class, target_class in DIRECTIONS:
            plan.append({
                "record_id": f"signed-weight-v5__{pair['pair_id']}__baseline__{direction}",
                "pair_id": pair["pair_id"],
                "axis": AXIS,
                "condition": "baseline",
                "direction": direction,
                "source_class": source_class,
                "target_class": target_class,
                "alpha": 0.0,
            })
            for alpha in ALPHAS:
                for method in METHODS:
                    plan.append({
                        "record_id": (
                            f"signed-weight-v5__{pair['pair_id']}__{method}__{direction}"
                            f"__alpha-{_alpha_label(alpha)}"
                        ),
                        "pair_id": pair["pair_id"],
                        "axis": AXIS,
                        "condition": "signed_weight_holdout",
                        "direction": direction,
                        "source_class": source_class,
                        "target_class": target_class,
                        "method": method,
                        "alpha": alpha,
                    })
    expected = len(pairs) * len(DIRECTIONS) * (1 + len(ALPHAS) * len(METHODS))
    ids = [row["record_id"] for row in plan]
    if len(plan) != expected or len(ids) != len(set(ids)):
        raise ValueError("plan count or ID uniqueness failure")
    return plan


def main() -> None:
    from .fit_subspaces_v4 import _load_temporal_codes

    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--operators", type=Path, required=True)
    parser.add_argument("--temporal-codes", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trajectories = read_jsonl(args.trajectories)
    trajectory_by_id = {str(row["record_id"]): row for row in trajectories}
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    holdout_count = len(fit["axes"][AXIS]["internal_holdout_pair_ids"])
    eligible_pairs = select_holdout_pairs(discovery, fit, holdout_count)
    record_ids, indices, values = _load_temporal_codes(args.temporal_codes)
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    with np.load(args.operators) as archive:
        features = archive[f"{AXIS}__features"].astype(np.int64).tolist()
        weights = archive[f"{AXIS}__weights"].astype(np.float32)
        decoder = archive[f"{AXIS}__decoder"].astype(np.float32)
        random_features = archive[f"{AXIS}__random_features"].astype(np.int64).tolist()
        random_decoder = archive[f"{AXIS}__random_decoder"].astype(np.float32)
    selected_ids = {
        str(pair[key]) for pair in eligible_pairs for key in ("low_record_id", "high_record_id")
    }
    missing = sorted(selected_ids.difference(record_index))
    if missing:
        raise ValueError(f"selected holdout records missing temporal codes: {missing}")
    pairs: list[dict[str, Any]] = []
    for pair in eligible_pairs:
        low_index = record_index[str(pair["low_record_id"])]
        high_index = record_index[str(pair["high_record_id"])]
        candidate = weighted_activation_update(
            indices[high_index], values[high_index],
            indices[low_index], values[low_index],
            features, decoder, weights,
        )
        equal = weighted_activation_update(
            indices[high_index], values[high_index],
            indices[low_index], values[low_index],
            features, decoder, np.ones_like(weights),
        )
        random = weighted_activation_update(
            indices[high_index], values[high_index],
            indices[low_index], values[low_index],
            random_features, random_decoder, weights,
        )
        if min(np.linalg.norm(candidate), np.linalg.norm(equal), np.linalg.norm(random)) > 0:
            pairs.append(pair)
        if len(pairs) == args.pairs:
            break
    if len(pairs) < args.pairs:
        raise ValueError(f"only {len(pairs)} holdout pairs have nonzero candidate/control updates")
    plan = build_plan(pairs)
    dry_run_norms: list[dict[str, Any]] = []
    for pair in pairs:
        low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
        for direction, source_class, _target_class in DIRECTIONS:
            source_id = high_id if source_class == "high" else low_id
            target_id = low_id if source_class == "high" else high_id
            source_index, target_index = record_index[source_id], record_index[target_id]
            candidate = weighted_activation_update(
                indices[source_index], values[source_index],
                indices[target_index], values[target_index],
                features, decoder, weights,
            )
            equal = weighted_activation_update(
                indices[source_index], values[source_index],
                indices[target_index], values[target_index],
                features, decoder, np.ones_like(weights),
            )
            random = weighted_activation_update(
                indices[source_index], values[source_index],
                indices[target_index], values[target_index],
                random_features, random_decoder, weights,
            )
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 0 or float(np.linalg.norm(equal)) <= 0 or float(np.linalg.norm(random)) <= 0:
                raise ValueError(f"zero control update for {pair['pair_id']}/{direction}")
            equal_matched, _ = norm_match(equal, candidate)
            random_matched, _ = norm_match(random, candidate)
            dry_run_norms.append({
                "pair_id": str(pair["pair_id"]),
                "direction": direction,
                "signed_weighted_l2": candidate_norm,
                "equal_weight_matched_l2": float(np.linalg.norm(equal_matched)),
                "random_weighted_matched_l2": float(np.linalg.norm(random_matched)),
            })
    audit = {
        "status": "dry_run_passed",
        "axis": AXIS,
        "selection_split": "train_internal_holdout",
        "fit_gate_passed": True,
        "development_accessed": False,
        "formal_evaluation_accessed": False,
        "selected_pair_ids": [str(pair["pair_id"]) for pair in pairs],
        "selected_case_ids": [str(pair["case_id"]) for pair in pairs],
        "pair_selection": "first pair IDs with nonzero candidate/equal/random activation updates",
        "features": features,
        "signed_weights": weights.tolist(),
        "random_features": random_features,
        "methods": list(METHODS),
        "alphas": list(ALPHAS),
        "expected_records": len(plan),
        "dry_run_norms": dry_run_norms,
        "evidence_boundary": (
            "Train-only internal-holdout generation smoke with model-assisted labels; "
            "not development or formal evaluation evidence."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dry_run_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return
    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    layer = _layers(model)[args.layer]
    plan_by_id = {row["record_id"]: row for row in plan}
    output_path = args.output / "signed_weight_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)

    for pair_index, pair in enumerate(pairs):
        seed = 2026080500 + pair_index
        low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
        for direction, source_class, target_class in DIRECTIONS:
            source_id = high_id if source_class == "high" else low_id
            target_id = low_id if target_class == "low" else high_id
            source_index, target_index = record_index[source_id], record_index[target_id]
            target_row = trajectory_by_id[target_id]
            baseline_id = f"signed-weight-v5__{pair['pair_id']}__baseline__{direction}"
            if baseline_id not in completed:
                text, state = _generate(
                    model,
                    tokenizer,
                    layer,
                    str(target_row["prompt"]),
                    str(target_row["response"]),
                    seed,
                    args.maximum_new_tokens,
                    None,
                )
                results.append({
                    **plan_by_id[baseline_id],
                    "case_id": pair["case_id"],
                    "source_record_id": source_id,
                    "target_record_id": target_id,
                    "seed": seed,
                    "prompt": target_row["prompt"],
                    "response": text,
                    **state,
                })
                completed.add(baseline_id)
                _atomic_jsonl(output_path, results)
            raw = {
                "signed_weighted": weighted_activation_update(
                    indices[source_index], values[source_index],
                    indices[target_index], values[target_index],
                    features, decoder, weights,
                ),
                "equal_weight": weighted_activation_update(
                    indices[source_index], values[source_index],
                    indices[target_index], values[target_index],
                    features, decoder, np.ones_like(weights),
                ),
                "random_weighted": weighted_activation_update(
                    indices[source_index], values[source_index],
                    indices[target_index], values[target_index],
                    random_features, random_decoder, weights,
                ),
            }
            matched = {"signed_weighted": raw["signed_weighted"]}
            scales = {"signed_weighted": 1.0}
            for method in ("equal_weight", "random_weighted"):
                matched[method], scales[method] = norm_match(
                    raw[method], raw["signed_weighted"]
                )
            for alpha in ALPHAS:
                for method in METHODS:
                    record_id = (
                        f"signed-weight-v5__{pair['pair_id']}__{method}__{direction}"
                        f"__alpha-{_alpha_label(alpha)}"
                    )
                    if record_id in completed:
                        continue
                    update = (matched[method] * np.float32(alpha)).astype(np.float32)
                    text, state = _generate(
                        model,
                        tokenizer,
                        layer,
                        str(target_row["prompt"]),
                        str(target_row["response"]),
                        seed,
                        args.maximum_new_tokens,
                        update,
                    )
                    results.append({
                        **plan_by_id[record_id],
                        "case_id": pair["case_id"],
                        "features": features if method != "random_weighted" else random_features,
                        "signed_weights": weights.tolist(),
                        "source_record_id": source_id,
                        "target_record_id": target_id,
                        "norm_match_scale": scales[method],
                        "seed": seed,
                        "prompt": target_row["prompt"],
                        "response": text,
                        **state,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"signed-weight {len(results)}/{len(plan)}", flush=True)

    manifest = {
        **audit,
        "type": "signed_relative_feature_weight_holdout_smoke_v5",
        "status": "ready_for_blind_scoring" if len(results) == len(plan) else "in_progress",
        "records": len(results),
        "model_assisted_scoring": "pending",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
