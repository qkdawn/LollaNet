from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .analyze_temporal_v3 import _feature_trajectory
from .contracts import AXES, assert_discovery_rows, read_jsonl
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom


def match_frequency_control(
    indices: np.ndarray, feature: int, excluded: set[int]
) -> tuple[int, int, int]:
    flat = indices.reshape(-1)
    valid = flat[flat >= 0]
    counts = np.bincount(valid, minlength=65536)
    target_count = int(counts[feature])
    eligible = np.flatnonzero(counts > 0)
    eligible = np.asarray([value for value in eligible if int(value) not in excluded])
    if not len(eligible):
        raise ValueError("no active random-control feature remains")
    control = min(
        eligible,
        key=lambda value: (
            abs(np.log1p(float(counts[value])) - np.log1p(float(target_count))),
            int(value),
        ),
    )
    return int(control), target_count, int(counts[control])


def match_frequency_control_for_pairs(
    indices: np.ndarray,
    values: np.ndarray,
    feature: int,
    excluded: set[int],
    record_index: dict[str, int],
    pairs: list[dict[str, Any]],
) -> tuple[int, int, int]:
    flat = indices.reshape(-1)
    valid = flat[flat >= 0]
    counts = np.bincount(valid, minlength=65536)
    target_count = int(counts[feature])
    nonzero_sets: list[set[int]] = []
    for pair in pairs:
        low_index = record_index[str(pair["low_record_id"])]
        high_index = record_index[str(pair["high_record_id"])]
        low: dict[int, float] = {}
        high: dict[int, float] = {}
        for candidate, activation in zip(
            indices[low_index].reshape(-1), values[low_index].reshape(-1)
        ):
            if candidate >= 0:
                low[int(candidate)] = low.get(int(candidate), 0.0) + float(activation)
        for candidate, activation in zip(
            indices[high_index].reshape(-1), values[high_index].reshape(-1)
        ):
            if candidate >= 0:
                high[int(candidate)] = high.get(int(candidate), 0.0) + float(activation)
        changed = {
            candidate for candidate in set(low) | set(high)
            if abs(high.get(candidate, 0.0) - low.get(candidate, 0.0)) > 1e-8
        }
        nonzero_sets.append(changed)
    eligible = set.intersection(*nonzero_sets) - excluded
    if not eligible:
        raise ValueError("no frequency control is nonzero on every engineering pair")
    control = min(
        eligible,
        key=lambda value: (
            abs(np.log1p(float(counts[value])) - np.log1p(float(target_count))),
            int(value),
        ),
    )
    return int(control), target_count, int(counts[control])


def _pair_delta(
    trajectory: np.ndarray, record_index: dict[str, int], pair: dict[str, Any]
) -> float:
    low = trajectory[record_index[str(pair["low_record_id"])]].mean()
    high = trajectory[record_index[str(pair["high_record_id"])]].mean()
    return float(high - low)


def select_engineering_pairs(
    discovery: dict[str, Any],
    indices: np.ndarray,
    values: np.ndarray,
    record_index: dict[str, int],
    axis: str,
    feature: int,
    count: int,
) -> list[dict[str, Any]]:
    feature_row = next(
        row for row in discovery["axes"][axis]["features"] if int(row["feature"]) == feature
    )
    expected_sign = np.sign(feature_row["paired_train"]["median_activation_delta"])
    trajectory = _feature_trajectory(indices, values, feature)
    candidates = []
    for pair in discovery["pairs"][axis]:
        if pair["split"] != "development":
            continue
        delta = _pair_delta(trajectory, record_index, pair)
        if delta != 0 and np.sign(delta) == expected_sign:
            candidates.append((abs(delta), str(pair["pair_id"]), pair))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    if len(candidates) < count:
        raise ValueError(f"axis {axis} has only {len(candidates)} nonzero aligned development pairs")
    return [row[2] for row in candidates[:count]]


def select_engineering_feature_and_pairs(
    discovery: dict[str, Any],
    temporal_analysis: dict[str, Any],
    indices: np.ndarray,
    values: np.ndarray,
    record_index: dict[str, int],
    axis: str,
    count: int,
) -> tuple[int, list[dict[str, Any]]]:
    for feature in temporal_analysis["axes"][axis]["early_available_candidates"]:
        try:
            pairs = select_engineering_pairs(
                discovery, indices, values, record_index, axis, int(feature), count
            )
        except ValueError:
            continue
        return int(feature), pairs
    raise ValueError(f"axis {axis} has no early candidate with {count} aligned development pairs")


def feature_swap_hook(
    direction: Any,
    source_values: np.ndarray,
    target_values: np.ndarray,
    response_start: int,
    coefficient_scale: float = 1.0,
):
    state = {"applied": False, "coefficient_l2": 0.0, "residual_update_l2": 0.0}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if state["applied"] or hidden.shape[1] <= 1:
            return output
        take = min(len(source_values), len(target_values), hidden.shape[1] - response_start)
        coefficients = torch.as_tensor(
            (source_values[:take] - target_values[:take]) * coefficient_scale,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        update = coefficients[:, None] * direction.to(
            device=hidden.device, dtype=hidden.dtype
        )[None, :]
        changed = hidden.clone()
        changed[0, response_start : response_start + take] += update
        state["applied"] = True
        state["coefficient_l2"] = float(torch.linalg.vector_norm(coefficients).float().cpu())
        state["residual_update_l2"] = float(torch.linalg.vector_norm(update).float().cpu())
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, state


def _generate_from_prefix(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    seed: int,
    maximum_new_tokens: int,
    layer: Any,
    direction: Any | None = None,
    source_values: np.ndarray | None = None,
    target_values: np.ndarray | None = None,
    coefficient_scale: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prefix_ids = response_ids[:, :16]
    combined = torch.cat((prompt_ids, prefix_ids), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    hook_state = {"applied": False, "coefficient_l2": 0.0, "residual_update_l2": 0.0}
    handle = None
    if direction is not None:
        hook, hook_state = feature_swap_hook(
            direction,
            np.asarray(source_values),
            np.asarray(target_values),
            int(prompt_ids.shape[1]),
            coefficient_scale,
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
    text = tokenizer.decode(
        output[0, int(prompt_ids.shape[1]) :], skip_special_tokens=True
    ).strip()
    if not text:
        raise RuntimeError("interchange generation produced an empty response")
    return text, hook_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--temporal-analysis", type=Path, required=True)
    parser.add_argument("--temporal-codes", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pairs-per-axis", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    trajectories = read_jsonl(args.trajectories)
    assert_discovery_rows(trajectories, expected=600)
    trajectory_by_id = {str(row["record_id"]): row for row in trajectories}
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    temporal_analysis = json.loads(args.temporal_analysis.read_text(encoding="utf-8"))
    with np.load(args.temporal_codes) as archive:
        record_ids = archive["record_ids"].tolist()
        indices = archive["indices"].copy()
        values = archive["values"].copy()
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    selected_pairs: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, int] = {}
    for axis in AXES:
        feature, pairs = select_engineering_feature_and_pairs(
            discovery,
            temporal_analysis,
            indices,
            values,
            record_index,
            axis,
            args.pairs_per_axis,
        )
        selected[axis] = feature
        selected_pairs[axis] = pairs
    excluded = {
        int(feature)
        for axis in AXES
        for feature in discovery["axes"][axis]["candidate_sets"]["union_for_temporal_attribution"]
    }
    controls = {
        axis: match_frequency_control_for_pairs(
            indices, values, feature, excluded, record_index, selected_pairs[axis]
        )
        for axis, feature in selected.items()
    }

    raw_sae = torch.load(args.sae, map_location="cpu", weights_only=True)
    decoder = raw_sae["W_dec"].float()
    if decoder.shape[0] != 4096:
        decoder = decoder.T.contiguous()
    directions = {
        feature: decoder[:, feature].clone()
        for feature in {*selected.values(), *(value[0] for value in controls.values())}
    }
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
    output_path = args.output / "interchange_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)

    for axis_index, axis in enumerate(AXES):
        candidate = selected[axis]
        control, candidate_frequency, control_frequency = controls[axis]
        pairs = selected_pairs[axis]
        for pair_index, pair in enumerate(pairs):
            seed = 2026073000 + axis_index * 100 + pair_index
            low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
            for target_name, target_id in (("low", low_id), ("high", high_id)):
                record_id = f"interchange-v3__{pair['pair_id']}__baseline-{target_name}"
                if record_id in completed:
                    continue
                target_row = trajectory_by_id[target_id]
                text, hook_state = _generate_from_prefix(
                    model,
                    tokenizer,
                    str(target_row["prompt"]),
                    str(target_row["response"]),
                    seed,
                    args.maximum_new_tokens,
                    layer,
                )
                results.append({
                    "record_id": record_id,
                    "pair_id": pair["pair_id"],
                    "case_id": pair["case_id"],
                    "axis": axis,
                    "prompt": target_row["prompt"],
                    "condition": "baseline",
                    "target_class": target_name,
                    "target_record_id": target_id,
                    "seed": seed,
                    "response": text,
                    **hook_state,
                })
                completed.add(record_id)
                _atomic_jsonl(output_path, results)
                print(f"interchange {len(results)}/36", flush=True)

            for random_control, feature in ((False, candidate), (True, control)):
                feature_values = _feature_trajectory(indices, values, feature)
                candidate_values = _feature_trajectory(indices, values, candidate)
                for source_name, source_id, target_name, target_id in (
                    ("high", high_id, "low", low_id),
                    ("low", low_id, "high", high_id),
                ):
                    record_id = (
                        f"interchange-v3__{pair['pair_id']}__feature-{feature}"
                        f"__random-{int(random_control)}__{source_name}-to-{target_name}"
                    )
                    if record_id in completed:
                        continue
                    target_row = trajectory_by_id[target_id]
                    coefficient_scale = 1.0
                    if random_control:
                        candidate_delta = (
                            candidate_values[record_index[source_id]]
                            - candidate_values[record_index[target_id]]
                        )
                        control_delta = (
                            feature_values[record_index[source_id]]
                            - feature_values[record_index[target_id]]
                        )
                        candidate_update_norm = float(np.linalg.norm(candidate_delta)) * float(
                            torch.linalg.vector_norm(directions[candidate]).item()
                        )
                        control_update_norm = float(np.linalg.norm(control_delta)) * float(
                            torch.linalg.vector_norm(directions[feature]).item()
                        )
                        if control_update_norm <= 0:
                            raise ValueError("matched control has zero pair-specific update norm")
                        coefficient_scale = candidate_update_norm / control_update_norm
                    text, hook_state = _generate_from_prefix(
                        model,
                        tokenizer,
                        str(target_row["prompt"]),
                        str(target_row["response"]),
                        seed,
                        args.maximum_new_tokens,
                        layer,
                        directions[feature],
                        feature_values[record_index[source_id]],
                        feature_values[record_index[target_id]],
                        coefficient_scale,
                    )
                    results.append({
                        "record_id": record_id,
                        "pair_id": pair["pair_id"],
                        "case_id": pair["case_id"],
                        "axis": axis,
                        "prompt": target_row["prompt"],
                        "condition": "interchange",
                        "candidate_feature": candidate,
                        "feature": feature,
                        "random_control": random_control,
                        "candidate_token_frequency": candidate_frequency,
                        "control_token_frequency": control_frequency,
                        "coefficient_scale": coefficient_scale,
                        "source_class": source_name,
                        "target_class": target_name,
                        "source_record_id": source_id,
                        "target_record_id": target_id,
                        "seed": seed,
                        "response": text,
                        **hook_state,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"interchange {len(results)}/36", flush=True)

    (args.output / "manifest.json").write_text(json.dumps({
        "type": "sae_interchange_engineering_smoke_v3",
        "status": "ready_for_blind_scoring" if len(results) == 36 else "in_progress",
        "records": len(results),
        "expected_records": 36,
        "selected_features": selected,
        "matched_frequency_controls": {
            axis: {
                "feature": values[0],
                "candidate_token_frequency": values[1],
                "control_token_frequency": values[2],
            }
            for axis, values in controls.items()
        },
        "pairs_per_axis": args.pairs_per_axis,
        "prefix_tokens": 16,
        "continuation_tokens": args.maximum_new_tokens,
        "pair_selection": "development nonzero aligned feature delta; engineering only",
        "evidence_boundary": (
            "Actuator engineering smoke only. Pair selection used development activations and "
            "cannot establish a causal feature or support a paper result."
        ),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
