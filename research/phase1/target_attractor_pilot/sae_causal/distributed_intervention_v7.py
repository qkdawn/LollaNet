from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import read_jsonl
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom
from .signed_weight_holdout_v5 import ALPHAS, _alpha_label


METHODS = ("distributed", "random")


def select_distinct_holdout_pairs(
    discovery: dict[str, Any], fit: dict[str, Any], axis: str, count: int
) -> list[dict[str, Any]]:
    allowed = set(fit["axes"][axis]["internal_holdout_pair_ids"])
    selected: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for pair in sorted(discovery["pairs"][axis], key=lambda row: str(row["pair_id"])):
        case_id = str(pair["case_id"])
        if (
            pair["split"] == "train" and str(pair["pair_id"]) in allowed
            and case_id not in seen_cases
        ):
            selected.append(pair)
            seen_cases.add(case_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"axis {axis} has only {len(selected)} distinct holdout cases")
    return selected


def build_plan(axis_pairs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for axis, pairs in axis_pairs.items():
        for pair in pairs:
            for direction, source_class, target_class in (
                ("high-to-low", "high", "low"),
                ("low-to-high", "low", "high"),
            ):
                base = {
                    "axis": axis, "pair_id": str(pair["pair_id"]),
                    "direction": direction, "source_class": source_class,
                    "target_class": target_class,
                }
                plan.append({
                    **base,
                    "record_id": f"distributed-v7__{axis}__{pair['pair_id']}__baseline__{direction}",
                    "condition": "baseline", "method": "none", "alpha": 0.0,
                })
                for alpha in ALPHAS:
                    for method in METHODS:
                        plan.append({
                            **base,
                            "record_id": (
                                f"distributed-v7__{axis}__{pair['pair_id']}__{method}"
                                f"__{direction}__alpha-{_alpha_label(alpha)}"
                            ),
                            "condition": "distributed_intervention",
                            "method": method, "alpha": alpha,
                        })
    expected = sum(len(pairs) for pairs in axis_pairs.values()) * 2 * (1 + len(ALPHAS) * 2)
    if len(plan) != expected or len({row["record_id"] for row in plan}) != expected:
        raise ValueError("distributed intervention plan is incomplete")
    return plan


def window_steering_hook(
    direction: np.ndarray, coefficient: float, start_step: int, stop_step: int
):
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12 or not 0 <= start_step < stop_step:
        raise ValueError("invalid window steering parameters")
    unit = unit / norm
    state = {
        "generation_steps_seen": 0,
        "intervention_steps": 0,
        "applied_update_energy": 0.0,
    }

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            return output
        step = int(state["generation_steps_seen"])
        state["generation_steps_seen"] += 1
        if not start_step <= step < stop_step:
            return output
        update = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype) * coefficient
        changed = hidden.clone()
        changed[0, 0] += update
        state["intervention_steps"] += 1
        state["applied_update_energy"] += float(coefficient) ** 2
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, state


def _generate(
    model: Any, tokenizer: Any, layer: Any, prompt: str, response: str, seed: int,
    maximum_new_tokens: int, direction: np.ndarray | None, coefficient: float,
    start_step: int, stop_step: int,
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
    if direction is None:
        state = {"generation_steps_seen": 0, "intervention_steps": 0, "applied_update_energy": 0.0}
    else:
        hook, state = window_steering_hook(direction, coefficient, start_step, stop_step)
        handle = layer.register_forward_hook(hook)
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=combined, attention_mask=torch.ones_like(combined),
                max_new_tokens=maximum_new_tokens, do_sample=True, temperature=0.9,
                top_p=0.95, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    state["applied_update_energy_l2"] = float(np.sqrt(state["applied_update_energy"]))
    text = tokenizer.decode(
        output[0, int(prompt_ids.shape[1]) :], skip_special_tokens=True
    ).strip()
    if not text:
        raise RuntimeError("distributed intervention produced an empty response")
    return text, state


def _reconstruct(packet: Path, mapping_path: Path) -> dict[str, dict[str, str]]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for row in read_jsonl(packet):
        record_id = str(mapping[str(row["item_id"])])
        result[record_id] = {"prompt": str(row["prompt"]), "response": str(row["response"])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--subspace-manifest", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-axis", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    subspaces = json.loads(args.subspace_manifest.read_text(encoding="utf-8"))
    axes = [axis for axis, value in subspaces["axes"].items() if value["passes_fit_oof_gate"]]
    axis_pairs = {
        axis: select_distinct_holdout_pairs(discovery, fit, axis, args.pairs_per_axis)
        for axis in axes
    }
    plan = build_plan(axis_pairs)
    audit = {
        "type": "distributed_causal_intervention_v7",
        "status": "dry_run_passed",
        "axes": axes,
        "selected_pairs": {axis: [str(row["pair_id"]) for row in pairs] for axis, pairs in axis_pairs.items()},
        "selection_rule": "all axes passing the predeclared fit-only OOF gate",
        "development_accessed": False,
        "formal_evaluation_accessed": False,
        "expected_records": len(plan),
        "methods": list(METHODS),
        "alphas": list(ALPHAS),
        "evidence_boundary": "Train internal-holdout causal smoke; not formal evaluation evidence.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dry_run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return
    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text_by_id = _reconstruct(args.packet, args.mapping)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    record_index = {str(row["record_id"]): index for index, row in enumerate(records)}
    residuals = np.load(args.residuals, mmap_mode="r")
    mask = np.load(args.mask, mmap_mode="r")
    with np.load(args.directions) as archive:
        direction_by_axis = {key: archive[key].astype(np.float32) for key in archive.files}
    require_commit_headroom(args.minimum_commit_headroom_gb)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    output_path = args.output / "distributed_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)
    plan_by_id = {str(row["record_id"]): row for row in plan}

    for axis_index, axis in enumerate(axes):
        selected = subspaces["axes"][axis]["selected_on_fit_only"]
        layer_index = int(selected["layer"])
        layer_position = int(selected["layer_position"])
        window_position = int(selected["window_position"])
        window_start, window_stop = [int(value) for value in selected["window"]]
        start_step, stop_step = max(0, window_start - 16), max(1, window_stop - 16)
        layer = _layers(model)[layer_index]
        for pair_index, pair in enumerate(axis_pairs[axis]):
            seed = 2026080800 + axis_index * 100 + pair_index
            low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
            for direction_name, source_id, target_id in (
                ("high-to-low", high_id, low_id), ("low-to-high", low_id, high_id)
            ):
                target = text_by_id[target_id]
                baseline_id = f"distributed-v7__{axis}__{pair['pair_id']}__baseline__{direction_name}"
                if baseline_id not in completed:
                    text, state = _generate(
                        model, tokenizer, layer, target["prompt"], target["response"], seed,
                        args.maximum_new_tokens, None, 0.0, start_step, stop_step,
                    )
                    results.append({
                        **plan_by_id[baseline_id], "case_id": pair["case_id"],
                        "source_record_id": source_id, "target_record_id": target_id,
                        "layer": layer_index, "token_window": [window_start, window_stop],
                        "seed": seed, "prompt": target["prompt"], "response": text, **state,
                    })
                    completed.add(baseline_id)
                    _atomic_jsonl(output_path, results)
                source_index, target_index = record_index[source_id], record_index[target_id]
                if not (
                    mask[source_index, layer_position, window_position]
                    and mask[target_index, layer_position, window_position]
                ):
                    raise ValueError(f"selected pair lacks residual window: {pair['pair_id']}")
                delta = (
                    residuals[source_index, layer_position, window_position].astype(np.float32)
                    - residuals[target_index, layer_position, window_position].astype(np.float32)
                )
                coefficient = float(delta @ direction_by_axis[axis])
                for alpha in ALPHAS:
                    for method in METHODS:
                        record_id = (
                            f"distributed-v7__{axis}__{pair['pair_id']}__{method}"
                            f"__{direction_name}__alpha-{_alpha_label(alpha)}"
                        )
                        if record_id in completed:
                            continue
                        applied_coefficient = coefficient * float(alpha)
                        text, state = _generate(
                            model, tokenizer, layer, target["prompt"], target["response"], seed,
                            args.maximum_new_tokens, direction_by_axis[
                                axis if method == "distributed" else f"{axis}__random"
                            ], applied_coefficient, start_step, stop_step,
                        )
                        results.append({
                            **plan_by_id[record_id], "case_id": pair["case_id"],
                            "source_record_id": source_id, "target_record_id": target_id,
                            "layer": layer_index, "token_window": [window_start, window_stop],
                            "base_projection_delta": coefficient,
                            "applied_coefficient": applied_coefficient,
                            "seed": seed, "prompt": target["prompt"], "response": text, **state,
                        })
                        completed.add(record_id)
                        _atomic_jsonl(output_path, results)
                        print(f"distributed-intervention {len(results)}/{len(plan)}", flush=True)
    manifest = {
        **audit,
        "status": "ready_for_blind_scoring" if len(results) == len(plan) else "in_progress",
        "records": len(results),
        "model_assisted_scoring": "pending",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
