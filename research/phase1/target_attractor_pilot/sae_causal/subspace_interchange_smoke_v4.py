from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import AXES, assert_discovery_rows, read_jsonl
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom


METHODS = ("sae", "sae_random", "das", "das_random")


def subspace_swap_update(
    source: np.ndarray, target: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    delta = np.asarray(source, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    space = np.asarray(basis, dtype=np.float32)
    return (delta @ space @ space.T).astype(np.float32)


def norm_match(update: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    update_norm = float(np.linalg.norm(update))
    reference_norm = float(np.linalg.norm(reference))
    if update_norm <= 1e-12:
        raise ValueError("random subspace produced a zero update")
    scale = reference_norm / update_norm
    return (np.asarray(update, dtype=np.float32) * scale).astype(np.float32), float(scale)


def swap_hook(update: np.ndarray, response_start: int):
    state = {"applied": False, "residual_update_l2": 0.0}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if state["applied"] or hidden.shape[1] <= 1:
            return output
        take = min(update.shape[0], hidden.shape[1] - response_start)
        delta = torch.as_tensor(
            update[:take], device=hidden.device, dtype=hidden.dtype
        )
        changed = hidden.clone()
        changed[0, response_start : response_start + take] += delta
        state["applied"] = True
        state["residual_update_l2"] = float(
            torch.linalg.vector_norm(delta).float().cpu()
        )
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, state


def _generate(
    model: Any,
    tokenizer: Any,
    layer: Any,
    prompt: str,
    response: str,
    seed: int,
    maximum_new_tokens: int,
    update: np.ndarray | None,
) -> tuple[str, dict[str, Any]]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prefix_ids = response_ids[:, :16]
    combined = torch.cat((prompt_ids, prefix_ids), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    state = {"applied": False, "residual_update_l2": 0.0}
    handle = None
    if update is not None:
        hook, state = swap_hook(update, int(prompt_ids.shape[1]))
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
        raise RuntimeError("subspace interchange produced an empty response")
    return text, state


def _development_pairs(discovery: dict[str, Any], axis: str, count: int) -> list[dict[str, Any]]:
    pairs = sorted(
        [pair for pair in discovery["pairs"][axis] if pair["split"] == "development"],
        key=lambda pair: str(pair["pair_id"]),
    )
    if len(pairs) < count:
        raise ValueError(f"axis {axis} has only {len(pairs)} development pairs")
    return pairs[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--residual-shards", type=Path, required=True)
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
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    with np.load(args.bases) as archive:
        bases = {key: archive[key].copy() for key in archive.files}
    for axis in AXES:
        ranks = {bases[f"{axis}__{method}"].shape[1] for method in METHODS}
        if ranks != {int(fit["axes"][axis]["rank"])}:
            raise ValueError(f"axis {axis} subspaces are not rank matched")

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
    output_path = args.output / "subspace_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)
    expected_records = len(AXES) * args.pairs_per_axis * (2 + 2 * len(METHODS))

    for axis_index, axis in enumerate(AXES):
        for pair_index, pair in enumerate(
            _development_pairs(discovery, axis, args.pairs_per_axis)
        ):
            seed = 2026073100 + axis_index * 100 + pair_index
            low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
            residuals = {
                record_id: np.load(args.residual_shards / f"{record_id}.npy")[:16]
                for record_id in (low_id, high_id)
            }
            for target_class, target_id in (("low", low_id), ("high", high_id)):
                record_id = f"subspace-v4__{pair['pair_id']}__baseline-{target_class}"
                if record_id not in completed:
                    target_row = trajectory_by_id[target_id]
                    text, state = _generate(
                        model, tokenizer, layer, str(target_row["prompt"]),
                        str(target_row["response"]), seed, args.maximum_new_tokens, None,
                    )
                    results.append({
                        "record_id": record_id, "pair_id": pair["pair_id"],
                        "case_id": pair["case_id"], "axis": axis,
                        "prompt": target_row["prompt"], "condition": "baseline",
                        "target_class": target_class, "target_record_id": target_id,
                        "seed": seed, "response": text, **state,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"subspace {len(results)}/{expected_records}", flush=True)

            for source_class, source_id, target_class, target_id in (
                ("high", high_id, "low", low_id),
                ("low", low_id, "high", high_id),
            ):
                raw_updates = {
                    method: subspace_swap_update(
                        residuals[source_id], residuals[target_id], bases[f"{axis}__{method}"]
                    )
                    for method in METHODS
                }
                updates = dict(raw_updates)
                scales = {method: 1.0 for method in METHODS}
                updates["sae_random"], scales["sae_random"] = norm_match(
                    raw_updates["sae_random"], raw_updates["sae"]
                )
                updates["das_random"], scales["das_random"] = norm_match(
                    raw_updates["das_random"], raw_updates["das"]
                )
                for method in METHODS:
                    record_id = (
                        f"subspace-v4__{pair['pair_id']}__{method}"
                        f"__{source_class}-to-{target_class}"
                    )
                    if record_id in completed:
                        continue
                    target_row = trajectory_by_id[target_id]
                    text, state = _generate(
                        model, tokenizer, layer, str(target_row["prompt"]),
                        str(target_row["response"]), seed, args.maximum_new_tokens,
                        updates[method],
                    )
                    results.append({
                        "record_id": record_id, "pair_id": pair["pair_id"],
                        "case_id": pair["case_id"], "axis": axis,
                        "prompt": target_row["prompt"], "condition": "subspace_interchange",
                        "method": method,
                        "rank": int(bases[f"{axis}__{method}"].shape[1]),
                        "source_class": source_class, "target_class": target_class,
                        "source_record_id": source_id, "target_record_id": target_id,
                        "norm_match_scale": scales[method], "seed": seed,
                        "response": text, **state,
                    })
                    completed.add(record_id)
                    _atomic_jsonl(output_path, results)
                    print(f"subspace {len(results)}/{expected_records}", flush=True)

    (args.output / "manifest.json").write_text(json.dumps({
        "type": "sae_vs_das_subspace_interchange_smoke_v4",
        "status": "ready_for_blind_scoring" if len(results) == expected_records else "in_progress",
        "records": len(results), "expected_records": expected_records,
        "pairs_per_axis": args.pairs_per_axis,
        "pair_selection": "deterministic first development pairs; no activation selection",
        "ranks": {axis: int(fit["axes"][axis]["rank"]) for axis in AXES},
        "methods": list(METHODS),
        "random_controls": "same rank and pairwise residual-update norm matched",
        "evidence_boundary": (
            "Two-pair development engineering smoke. Passing can only unlock a broader "
            "development test and cannot establish a causal subspace."
        ),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
