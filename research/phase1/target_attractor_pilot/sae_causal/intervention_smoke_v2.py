from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .causal import steering_hook
from .contracts import assert_discovery_rows, read_jsonl
from .resources import require_commit_headroom


AXES = ("specificity", "independence", "non_template")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_codes(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path) as archive:
        return (
            archive["indices"].copy(), archive["values"].copy(),
            archive["record_ids"].tolist(),
        )


def _feature_scale(indices: np.ndarray, values: np.ndarray, feature: int, train_count: int) -> float:
    dense = np.where(indices[:train_count] == feature, values[:train_count], 0.0).sum(axis=1)
    scale = float(np.std(dense, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"feature {feature} has no finite train activation scale")
    return scale


def _select_schedule(candidates: dict[str, Any], features_per_axis: int, controls_per_feature: int) -> list[dict[str, Any]]:
    schedule = []
    for axis in AXES:
        for row in candidates["candidates"][axis][:features_per_axis]:
            candidate = int(row["feature"])
            schedule.append({
                "axis": axis, "candidate_feature": candidate,
                "feature": candidate, "random_control": False,
            })
            controls = candidates["matched_random_controls"][str(candidate)][:controls_per_feature]
            for control in controls:
                schedule.append({
                    "axis": axis, "candidate_feature": candidate,
                    "feature": int(control), "random_control": True,
                })
    return schedule


def _development_prompts(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected, seen = [], set()
    for row in rows:
        if row["split"] != "development" or row["case_id"] in seen:
            continue
        seen.add(row["case_id"])
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} unique development prompts, found {len(selected)}")
    return selected


def _generate(model: Any, tokenizer: Any, prompt: str, seed: int, maximum_new_tokens: int) -> str:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.get_input_embeddings().weight.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=maximum_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("intervention generation produced an empty response")
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--features-per-axis", type=int, default=1)
    parser.add_argument("--controls-per-feature", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--strengths", type=float, nargs="+", default=[2.0])
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--steer-tokens", type=int, default=32)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=12.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    trajectories = read_jsonl(args.trajectories)
    assert_discovery_rows(trajectories, expected=600)
    prompt_rows = _development_prompts(trajectories, args.prompts)
    candidate_data = json.loads(args.candidates.read_text(encoding="utf-8"))
    if candidate_data.get("type") != "sae_association_candidates_v2":
        raise ValueError("intervention smoke requires train-only v2 candidates")
    schedule = _select_schedule(candidate_data, args.features_per_axis, args.controls_per_feature)
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
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    layer = _layers(model)[args.layer]
    output_path = args.output / "intervention_trajectories.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)

    for prompt_index, row in enumerate(prompt_rows):
        seed = 2026072700 + prompt_index
        baseline_id = f"smoke__{row['case_id']}__baseline"
        if baseline_id not in completed:
            response = _generate(model, tokenizer, str(row["prompt"]), seed, args.maximum_new_tokens)
            results.append({
                "record_id": baseline_id, "case_id": row["case_id"], "prompt": row["prompt"],
                "condition": "baseline", "seed": seed, "response": response,
            })
            completed.add(baseline_id)
            _atomic_jsonl(output_path, results)
        for condition in schedule:
            feature = int(condition["feature"])
            for strength in args.strengths:
                for sign in (-1, 1):
                    signed_strength = float(sign * strength)
                    record_id = (
                        f"smoke__{row['case_id']}__{condition['axis']}__candidate-{condition['candidate_feature']}"
                        f"__feature-{feature}__random-{int(condition['random_control'])}__dose-{signed_strength:+g}"
                    )
                    if record_id in completed:
                        continue
                    coefficient = signed_strength * scales[feature]
                    handle = layer.register_forward_hook(steering_hook(
                        directions[feature], coefficient, token_start=0, token_stop=args.steer_tokens,
                    ))
                    try:
                        response = _generate(model, tokenizer, str(row["prompt"]), seed, args.maximum_new_tokens)
                    finally:
                        handle.remove()
                    results.append({
                        "record_id": record_id, "case_id": row["case_id"], "prompt": row["prompt"],
                        **condition, "condition": "intervention", "signed_strength": signed_strength,
                        "feature_train_scale": scales[feature], "coefficient": coefficient,
                        "seed": seed, "response": response,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"generated {len(results)}", flush=True)

    (args.output / "manifest.json").write_text(json.dumps({
        "type": "sae_causal_intervention_engineering_smoke_v2",
        "status": "ready_for_blind_scoring",
        "records": len(results),
        "prompts": len(prompt_rows),
        "features_per_axis": args.features_per_axis,
        "controls_per_feature": args.controls_per_feature,
        "strengths": args.strengths,
        "evidence_boundary": "Engineering smoke only; not a causal-feature result.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
