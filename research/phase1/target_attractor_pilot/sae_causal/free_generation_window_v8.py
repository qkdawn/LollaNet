from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import read_jsonl
from .distributed_intervention_v7 import _reconstruct, select_distinct_holdout_pairs
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom


METHODS = ("distributed", "random")
POLICIES = ("early", "matched", "full")
ALPHA = 1.0


def free_generation_hook(
    direction: np.ndarray, coefficient: float, start_step: int, stop_step: int
):
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12 or not 0 <= start_step < stop_step:
        raise ValueError("invalid free-generation steering parameters")
    unit /= norm
    state = {"step": 0, "initialized": False, "intervention_steps": 0, "applied_update_energy": 0.0}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if not state["initialized"]:
            state["initialized"] = True
            step, target = 0, -1
        elif hidden.shape[1] == 1:
            state["step"] += 1
            step, target = int(state["step"]), 0
        else:
            return output
        if not start_step <= step < stop_step:
            return output
        update = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype) * coefficient
        changed = hidden.clone()
        changed[0, target] += update
        state["intervention_steps"] += 1
        state["applied_update_energy"] += float(coefficient) ** 2
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, state


def build_plan(axis_pairs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for axis, pairs in axis_pairs.items():
        for pair in pairs:
            plan.append({
                "record_id": f"free-v8__{axis}__{pair['pair_id']}__baseline",
                "axis": axis, "pair_id": str(pair["pair_id"]),
                "condition": "baseline", "policy": "none", "method": "none",
            })
            for direction in ("high-to-low", "low-to-high"):
                for policy in POLICIES:
                    for method in METHODS:
                        plan.append({
                            "record_id": (
                                f"free-v8__{axis}__{pair['pair_id']}__{direction}"
                                f"__{policy}__{method}"
                            ),
                            "axis": axis, "pair_id": str(pair["pair_id"]),
                            "condition": "free_generation_window", "direction": direction,
                            "policy": policy, "method": method, "alpha": ALPHA,
                        })
    expected = sum(len(pairs) for pairs in axis_pairs.values()) * 13
    if len(plan) != expected or len({row["record_id"] for row in plan}) != expected:
        raise ValueError("free-generation plan is incomplete")
    return plan


def _generate(
    model: Any, tokenizer: Any, layer: Any, prompt: str, seed: int,
    maximum_new_tokens: int, direction: np.ndarray | None, coefficient: float,
    start_step: int, stop_step: int,
) -> tuple[str, dict[str, Any]]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.get_input_embeddings().weight.device)
    handle = None
    if direction is None:
        state = {"intervention_steps": 0, "applied_update_energy": 0.0}
    else:
        hook, state = free_generation_hook(direction, coefficient, start_step, stop_step)
        handle = layer.register_forward_hook(hook)
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=maximum_new_tokens, do_sample=True, temperature=0.9,
                top_p=0.95, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    state["applied_update_energy_l2"] = float(np.sqrt(state["applied_update_energy"]))
    text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("free generation produced an empty response")
    return text, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--subspace-manifest", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-axis", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=96)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    subspaces = json.loads(args.subspace_manifest.read_text(encoding="utf-8"))
    axes = [axis for axis, value in subspaces["axes"].items() if value["passes_fit_oof_gate"]]
    axis_pairs = {axis: select_distinct_holdout_pairs(discovery, fit, axis, args.pairs_per_axis) for axis in axes}
    plan = build_plan(axis_pairs)
    audit = {
        "type": "free_generation_temporal_window_v8", "status": "dry_run_passed",
        "axes": axes, "policies": list(POLICIES), "methods": list(METHODS),
        "alpha": ALPHA, "expected_records": len(plan),
        "selected_pairs": {axis: [str(row["pair_id"]) for row in pairs] for axis, pairs in axis_pairs.items()},
        "response_prefix_tokens": 0, "development_accessed": False,
        "formal_evaluation_accessed": False,
        "evidence_boundary": "Train internal-holdout free-generation diagnostic only.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dry_run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(audit, indent=2)); return
    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text_by_id = _reconstruct(args.packet, args.mapping)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    record_index = {str(row["record_id"]): index for index, row in enumerate(records)}
    residuals = np.load(args.residuals, mmap_mode="r")
    with np.load(args.directions) as archive:
        directions = {key: archive[key].astype(np.float32) for key in archive.files}
    require_commit_headroom(args.minimum_commit_headroom_gb)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    output_path = args.output / "free_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)
    plan_by_id = {str(row["record_id"]): row for row in plan}
    for axis_index, axis in enumerate(axes):
        selected = subspaces["axes"][axis]["selected_on_fit_only"]
        layer = _layers(model)[int(selected["layer"])]
        layer_position, window_position = int(selected["layer_position"]), int(selected["window_position"])
        matched_window = tuple(int(value) for value in selected["window"])
        windows = {"early": (0, 8), "matched": matched_window, "full": (0, args.maximum_new_tokens)}
        for pair_index, pair in enumerate(axis_pairs[axis]):
            seed = 2026080900 + axis_index * 100 + pair_index
            low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
            prompt = text_by_id[low_id]["prompt"]
            if prompt != text_by_id[high_id]["prompt"]:
                raise ValueError("matched pair prompts differ")
            baseline_id = f"free-v8__{axis}__{pair['pair_id']}__baseline"
            if baseline_id not in completed:
                text, state = _generate(model, tokenizer, layer, prompt, seed, args.maximum_new_tokens, None, 0.0, 0, 1)
                results.append({**plan_by_id[baseline_id], "case_id": pair["case_id"], "seed": seed, "prompt": prompt, "response": text, **state})
                completed.add(baseline_id); _atomic_jsonl(output_path, results)
            for direction_name, source_id, target_id in (
                ("high-to-low", high_id, low_id), ("low-to-high", low_id, high_id)
            ):
                delta = (
                    residuals[record_index[source_id], layer_position, window_position].astype(np.float32)
                    - residuals[record_index[target_id], layer_position, window_position].astype(np.float32)
                )
                coefficient = float(delta @ directions[axis])
                for policy in POLICIES:
                    start, stop = windows[policy]
                    for method in METHODS:
                        record_id = f"free-v8__{axis}__{pair['pair_id']}__{direction_name}__{policy}__{method}"
                        if record_id in completed:
                            continue
                        text, state = _generate(
                            model, tokenizer, layer, prompt, seed, args.maximum_new_tokens,
                            directions[axis if method == "distributed" else f"{axis}__random"],
                            coefficient, start, stop,
                        )
                        results.append({
                            **plan_by_id[record_id], "case_id": pair["case_id"],
                            "source_record_id": source_id, "target_record_id": target_id,
                            "layer": int(selected["layer"]), "injection_window": [start, stop],
                            "base_projection_delta": coefficient, "applied_coefficient": coefficient,
                            "seed": seed, "prompt": prompt, "response": text, **state,
                        })
                        completed.add(record_id); _atomic_jsonl(output_path, results)
                        print(f"free-window {len(results)}/{len(plan)}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({
        **audit, "status": "ready_for_blind_scoring" if len(results) == len(plan) else "in_progress",
        "records": len(results), "model_assisted_scoring": "pending",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
