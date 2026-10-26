from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers

from .causal import steering_hook
from .contracts import assert_discovery_rows, read_jsonl
from .intervention_smoke_v2 import (
    AXES,
    _atomic_jsonl,
    _development_prompts,
    _feature_scale,
    _generate,
    _load_codes,
)
from .resources import require_commit_headroom


def _select_prescreen_schedule(
    candidates: dict[str, Any], features_per_axis: int, controls_per_feature: int
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for axis in AXES:
        axis_candidates = candidates["candidates"][axis][:features_per_axis]
        if len(axis_candidates) != features_per_axis:
            raise ValueError(f"axis {axis} has only {len(axis_candidates)} candidates")
        for candidate_row in axis_candidates:
            candidate = int(candidate_row["feature"])
            train_rho = float(candidate_row["train_rho"])
            common = {
                "axis": axis,
                "candidate_feature": candidate,
                "candidate_train_rho": train_rho,
                "target_improving_feature_sign": 1 if train_rho > 0 else -1,
            }
            schedule.append({
                **common,
                "feature": candidate,
                "random_control": False,
                "control_index": None,
            })
            controls = candidates["matched_random_controls"][str(candidate)][:controls_per_feature]
            if len(controls) != controls_per_feature:
                raise ValueError(f"feature {candidate} has only {len(controls)} controls")
            for control_index, control in enumerate(controls):
                schedule.append({
                    **common,
                    "feature": int(control),
                    "random_control": True,
                    "control_index": control_index,
                })
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--features-per-axis", type=int, default=4)
    parser.add_argument("--controls-per-feature", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--strengths", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--maximum-new-tokens", type=int, default=384)
    parser.add_argument("--steer-tokens", type=int, default=32)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=12.0)
    parser.add_argument("--record-limit", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    trajectories = read_jsonl(args.trajectories)
    assert_discovery_rows(trajectories, expected=600)
    prompt_rows = _development_prompts(trajectories, args.prompts)
    candidate_data = json.loads(args.candidates.read_text(encoding="utf-8"))
    if candidate_data.get("type") != "sae_association_candidates_v2":
        raise ValueError("prescreen requires corrected train-only v2 candidates")
    if candidate_data.get("selection_split") != "train":
        raise ValueError("candidate selection must be train-only")

    schedule = _select_prescreen_schedule(
        candidate_data, args.features_per_axis, args.controls_per_feature
    )
    expected_records = args.prompts * (1 + len(schedule) * len(args.strengths) * 2)
    indices, values, record_ids = _load_codes(args.codes)
    if record_ids != [str(row["record_id"]) for row in trajectories]:
        raise ValueError("codes do not match frozen trajectory order")
    train_count = sum(row["split"] == "train" for row in trajectories)
    features = sorted({int(row["feature"]) for row in schedule})
    scales = {feature: _feature_scale(indices, values, feature, train_count) for feature in features}

    raw_sae = torch.load(args.sae, map_location="cpu", weights_only=True)
    decoder = raw_sae["W_dec"].float()
    directions = {feature: decoder[:, feature].clone() for feature in features}
    del raw_sae, decoder
    gc.collect()

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

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "intervention_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)
    generated_this_run = 0

    for prompt_index, row in enumerate(prompt_rows):
        seed = 2026072800 + prompt_index
        baseline_id = f"prescreen-v2__{row['case_id']}__baseline"
        if baseline_id not in completed:
            response = _generate(model, tokenizer, str(row["prompt"]), seed, args.maximum_new_tokens)
            results.append({
                "record_id": baseline_id,
                "case_id": row["case_id"],
                "prompt": row["prompt"],
                "condition": "baseline",
                "seed": seed,
                "response": response,
            })
            completed.add(baseline_id)
            generated_this_run += 1
            _atomic_jsonl(output_path, results)
            print(f"generated {len(results)}/{expected_records}", flush=True)
            if args.record_limit is not None and generated_this_run >= args.record_limit:
                break

        stop = False
        for condition in schedule:
            feature = int(condition["feature"])
            for strength in args.strengths:
                for sign in (-1, 1):
                    signed_strength = float(sign * strength)
                    record_id = (
                        f"prescreen-v2__{row['case_id']}__{condition['axis']}"
                        f"__candidate-{condition['candidate_feature']}__feature-{feature}"
                        f"__random-{int(condition['random_control'])}__dose-{signed_strength:+g}"
                    )
                    if record_id in completed:
                        continue
                    coefficient = signed_strength * scales[feature]
                    handle = layer.register_forward_hook(
                        steering_hook(
                            directions[feature],
                            coefficient,
                            token_start=0,
                            token_stop=args.steer_tokens,
                        )
                    )
                    try:
                        response = _generate(
                            model, tokenizer, str(row["prompt"]), seed, args.maximum_new_tokens
                        )
                    finally:
                        handle.remove()
                    results.append({
                        "record_id": record_id,
                        "case_id": row["case_id"],
                        "prompt": row["prompt"],
                        **condition,
                        "condition": "intervention",
                        "signed_strength": signed_strength,
                        "feature_train_scale": scales[feature],
                        "coefficient": coefficient,
                        "seed": seed,
                        "response": response,
                    })
                    completed.add(record_id)
                    generated_this_run += 1
                    _atomic_jsonl(output_path, results)
                    print(f"generated {len(results)}/{expected_records}", flush=True)
                    if args.record_limit is not None and generated_this_run >= args.record_limit:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    status = "ready_for_blind_scoring" if len(results) == expected_records else "in_progress"
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "type": "sae_causal_adaptive_prescreen_v2",
                "status": status,
                "records": len(results),
                "expected_records": expected_records,
                "prompts": args.prompts,
                "features_per_axis": args.features_per_axis,
                "controls_per_feature": args.controls_per_feature,
                "strengths": args.strengths,
                "maximum_new_tokens": args.maximum_new_tokens,
                "candidate_selection": "train_only",
                "split": "development",
                "evidence_boundary": (
                    "Discovery prescreen only. Passing candidates require expanded random controls "
                    "and the charter's 40-prompt confirmation before a causal claim."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
