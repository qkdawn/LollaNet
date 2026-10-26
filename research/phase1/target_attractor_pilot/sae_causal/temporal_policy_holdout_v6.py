from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import read_jsonl
from .fit_signed_weights_v5 import weighted_activation_update
from .fit_subspaces_v4 import _load_temporal_codes
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom
from .signed_dose_train_v4 import DIRECTIONS
from .signed_weight_holdout_v5 import (
    ALPHAS,
    AXIS,
    METHODS,
    _alpha_label,
    select_holdout_pairs,
)
from .subspace_interchange_smoke_v4 import norm_match, swap_hook


POLICY_METHODS = (
    ("prefix_once", "signed_weighted"),
    ("persistent", "signed_weighted"),
    ("persistent", "equal_weight"),
    ("persistent", "random_weighted"),
    ("closed_loop", "signed_weighted"),
    ("closed_loop", "random_weighted"),
)


def build_plan(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for pair in pairs:
        for direction, source_class, target_class in DIRECTIONS:
            base = {
                "pair_id": str(pair["pair_id"]),
                "axis": AXIS,
                "direction": direction,
                "source_class": source_class,
                "target_class": target_class,
            }
            plan.append({
                **base,
                "record_id": f"temporal-v6__{pair['pair_id']}__baseline__{direction}",
                "condition": "baseline",
                "policy": "none",
                "alpha": 0.0,
            })
            for alpha in ALPHAS:
                for policy, method in POLICY_METHODS:
                    plan.append({
                        **base,
                        "record_id": (
                            f"temporal-v6__{pair['pair_id']}__{policy}__{method}"
                            f"__{direction}__alpha-{_alpha_label(alpha)}"
                        ),
                        "condition": "temporal_policy_holdout",
                        "policy": policy,
                        "method": method,
                        "alpha": alpha,
                    })
    expected = len(pairs) * len(DIRECTIONS) * (1 + len(ALPHAS) * len(POLICY_METHODS))
    ids = [row["record_id"] for row in plan]
    if len(plan) != expected or len(ids) != len(set(ids)):
        raise ValueError("temporal-policy plan count or ID uniqueness failure")
    return plan


def _fixed_direction(update: np.ndarray) -> np.ndarray:
    direction = np.asarray(update, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("temporal mean direction is zero")
    return (direction / norm).astype(np.float32)


def temporal_hook(
    direction: np.ndarray,
    policy: str,
    energy_budget: float,
    maximum_steps: int,
    source_anchor: float | None = None,
    target_anchor: float | None = None,
    alpha: float = 1.0,
):
    if policy not in {"persistent", "closed_loop"}:
        raise ValueError(f"unknown temporal policy: {policy}")
    if energy_budget <= 0 or maximum_steps <= 0:
        raise ValueError("temporal intervention requires positive budget and steps")
    unit = _fixed_direction(np.asarray(direction, dtype=np.float32).reshape(1, -1))
    per_step_cap = float(energy_budget) / float(np.sqrt(maximum_steps))
    if policy == "closed_loop" and (source_anchor is None or target_anchor is None):
        raise ValueError("closed_loop requires source and target anchors")
    desired_projection = None
    if policy == "closed_loop":
        desired_projection = float(target_anchor + alpha * (source_anchor - target_anchor))
    state: dict[str, Any] = {
        "generation_steps_seen": 0,
        "intervention_steps": 0,
        "applied_update_l2": 0.0,
        "applied_update_energy": 0.0,
        "energy_budget": float(energy_budget),
        "per_step_cap": per_step_cap,
        "desired_projection": desired_projection,
    }

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        # The prefill has sequence length > 1. Only intervene in cached single-token steps.
        if hidden.shape[1] != 1 or state["generation_steps_seen"] >= maximum_steps:
            return output
        state["generation_steps_seen"] += 1
        unit_tensor = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
        magnitude = per_step_cap
        if policy == "closed_loop":
            projection = float(
                torch.dot(hidden[0, 0].float(), unit_tensor.float()).detach().cpu()
            )
            error = float(desired_projection) - projection
            magnitude = max(-per_step_cap, min(per_step_cap, error))
        if abs(magnitude) <= 1e-12:
            return output
        update = unit_tensor * magnitude
        changed = hidden.clone()
        changed[0, 0] += update
        state["intervention_steps"] += 1
        state["applied_update_l2"] += abs(float(magnitude))
        state["applied_update_energy"] += float(magnitude) ** 2
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, state


def _anchor(residual: np.ndarray, direction: np.ndarray) -> float:
    unit = _fixed_direction(np.asarray(direction, dtype=np.float32).reshape(1, -1))
    values = np.asarray(residual, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != unit.shape[0]:
        raise ValueError("residual anchor shape mismatch")
    tail = values[max(0, values.shape[0] // 2) :]
    return float(np.mean(tail @ unit))


def _generate_temporal(
    model: Any,
    tokenizer: Any,
    layer: Any,
    prompt: str,
    response: str,
    seed: int,
    maximum_new_tokens: int,
    policy: str,
    update: np.ndarray | None,
    source_residual: np.ndarray | None = None,
    target_residual: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prefix_ids = response_ids[:, :16]
    combined = torch.cat((prompt_ids, prefix_ids), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    handle = None
    if update is None:
        state: dict[str, Any] = {
            "generation_steps_seen": 0,
            "intervention_steps": 0,
            "applied_update_l2": 0.0,
            "applied_update_energy": 0.0,
            "energy_budget": 0.0,
        }
    else:
        energy_budget = float(np.linalg.norm(update))
        if policy == "prefix_once":
            hook, legacy_state = swap_hook(update, int(prompt_ids.shape[1]))
            state = {
                "generation_steps_seen": 0,
                "intervention_steps": 1,
                "applied_update_l2": energy_budget,
                "applied_update_energy": energy_budget**2,
                "energy_budget": energy_budget,
                "legacy_state": legacy_state,
            }
        else:
            fixed = _fixed_direction(update)
            kwargs: dict[str, Any] = {}
            if policy == "closed_loop":
                if source_residual is None or target_residual is None:
                    raise ValueError("closed_loop residual anchors are missing")
                kwargs = {
                    "source_anchor": _anchor(source_residual, fixed),
                    "target_anchor": _anchor(target_residual, fixed),
                    # Alpha is already included in update norm; anchors use the donor endpoint.
                    "alpha": 1.0,
                }
            hook, state = temporal_hook(
                fixed,
                policy,
                energy_budget,
                maximum_new_tokens,
                **kwargs,
            )
        handle = layer.register_forward_hook(hook)
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=combined,
                attention_mask=torch.ones_like(combined),
                max_new_tokens=maximum_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    if "legacy_state" in state:
        legacy = state.pop("legacy_state")
        state["intervention_steps"] = int(bool(legacy["applied"]))
        state["applied_update_l2"] = float(legacy["residual_update_l2"])
        state["applied_update_energy"] = float(legacy["residual_update_l2"]) ** 2
    state["applied_update_energy"] = float(state["applied_update_energy"])
    state["applied_update_energy_l2"] = float(np.sqrt(state["applied_update_energy"]))
    text = tokenizer.decode(
        output[0, int(prompt_ids.shape[1]) :], skip_special_tokens=True
    ).strip()
    if not text:
        raise RuntimeError("temporal policy produced an empty response")
    return text, state


def _baseline_targets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "baseline":
            continue
        target_id = str(row["target_record_id"])
        current = targets.get(target_id)
        if current is not None and (
            current["prompt"] != row["prompt"] or current["response"] != row["response"]
        ):
            raise ValueError(f"inconsistent baseline target text for {target_id}")
        targets[target_id] = row
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--operators", type=Path, required=True)
    parser.add_argument("--temporal-codes", type=Path, required=True)
    parser.add_argument("--residual-shards", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_trajectories)
    baseline_targets = _baseline_targets(source_rows)
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    eligible = select_holdout_pairs(
        discovery, fit, len(fit["axes"][AXIS]["internal_holdout_pair_ids"])
    )
    record_ids, indices, values = _load_temporal_codes(args.temporal_codes)
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    with np.load(args.operators) as archive:
        features = archive[f"{AXIS}__features"].astype(np.int64).tolist()
        weights = archive[f"{AXIS}__weights"].astype(np.float32)
        decoder = archive[f"{AXIS}__decoder"].astype(np.float32)
        random_features = archive[f"{AXIS}__random_features"].astype(np.int64).tolist()
        random_decoder = archive[f"{AXIS}__random_decoder"].astype(np.float32)

    pairs: list[dict[str, Any]] = []
    for pair in eligible:
        required = {str(pair["low_record_id"]), str(pair["high_record_id"])}
        if required <= baseline_targets.keys() and required <= record_index.keys():
            pairs.append(pair)
        if len(pairs) == args.pairs:
            break
    if len(pairs) != args.pairs:
        raise ValueError("source trajectories do not cover enough eligible holdout pairs")
    plan = build_plan(pairs)
    audit = {
        "type": "temporal_policy_holdout_v6",
        "status": "dry_run_passed",
        "axis": AXIS,
        "selection_split": "train_internal_holdout",
        "development_accessed": False,
        "formal_evaluation_accessed": False,
        "selected_pair_ids": [str(pair["pair_id"]) for pair in pairs],
        "policies_and_methods": [list(row) for row in POLICY_METHODS],
        "alphas": list(ALPHAS),
        "expected_records": len(plan),
        "budget_matching": (
            "Within each pair/direction/alpha, methods share the signed prefix update L2 "
            "as a maximum trajectory energy budget; actual energy is recorded per output."
        ),
        "closed_loop_target": (
            "Late-half mean donor residual projection along the fixed intervention direction."
        ),
        "evidence_boundary": (
            "Train-only internal-holdout temporal-policy smoke. This cannot select on "
            "development or support a formal evaluation claim."
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
    output_path = args.output / "temporal_policy_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)
    plan_by_id = {str(row["record_id"]): row for row in plan}

    for pair_index, pair in enumerate(pairs):
        seed = 2026080600 + pair_index
        low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
        for direction, source_class, target_class in DIRECTIONS:
            source_id = high_id if source_class == "high" else low_id
            target_id = low_id if target_class == "low" else high_id
            source_index, target_index = record_index[source_id], record_index[target_id]
            target_row = baseline_targets[target_id]
            source_residual = np.load(args.residual_shards / f"{source_id}.npy")
            target_residual = np.load(args.residual_shards / f"{target_id}.npy")
            baseline_id = f"temporal-v6__{pair['pair_id']}__baseline__{direction}"
            if baseline_id not in completed:
                text, state = _generate_temporal(
                    model, tokenizer, layer, str(target_row["prompt"]),
                    str(target_row["response"]), seed, args.maximum_new_tokens,
                    "none", None,
                )
                results.append({
                    **plan_by_id[baseline_id], "case_id": pair["case_id"],
                    "source_record_id": source_id, "target_record_id": target_id,
                    "seed": seed, "prompt": target_row["prompt"], "response": text,
                    **state,
                })
                completed.add(baseline_id)
                _atomic_jsonl(output_path, results)
            raw = {
                "signed_weighted": weighted_activation_update(
                    indices[source_index], values[source_index], indices[target_index],
                    values[target_index], features, decoder, weights,
                ),
                "equal_weight": weighted_activation_update(
                    indices[source_index], values[source_index], indices[target_index],
                    values[target_index], features, decoder, np.ones_like(weights),
                ),
                "random_weighted": weighted_activation_update(
                    indices[source_index], values[source_index], indices[target_index],
                    values[target_index], random_features, random_decoder, weights,
                ),
            }
            matched = {"signed_weighted": raw["signed_weighted"]}
            scales = {"signed_weighted": 1.0}
            for method in METHODS[1:]:
                matched[method], scales[method] = norm_match(
                    raw[method], raw["signed_weighted"]
                )
            for alpha in ALPHAS:
                for policy, method in POLICY_METHODS:
                    record_id = (
                        f"temporal-v6__{pair['pair_id']}__{policy}__{method}"
                        f"__{direction}__alpha-{_alpha_label(alpha)}"
                    )
                    if record_id in completed:
                        continue
                    update = (matched[method] * np.float32(alpha)).astype(np.float32)
                    text, state = _generate_temporal(
                        model, tokenizer, layer, str(target_row["prompt"]),
                        str(target_row["response"]), seed, args.maximum_new_tokens,
                        policy, update, source_residual, target_residual,
                    )
                    results.append({
                        **plan_by_id[record_id], "case_id": pair["case_id"],
                        "features": features if method != "random_weighted" else random_features,
                        "signed_weights": weights.tolist(),
                        "source_record_id": source_id, "target_record_id": target_id,
                        "norm_match_scale": scales[method], "seed": seed,
                        "prompt": target_row["prompt"], "response": text, **state,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"temporal-policy {len(results)}/{len(plan)}", flush=True)

    manifest = {
        **audit,
        "status": "ready_for_blind_scoring" if len(results) == len(plan) else "in_progress",
        "records": len(results),
        "model_assisted_scoring": "pending",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
