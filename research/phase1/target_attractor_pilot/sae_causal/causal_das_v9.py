from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import read_jsonl
from .distributed_intervention_v7 import _reconstruct
from .extract_distributed_residuals_v7 import TOKEN_WINDOWS
from .fit_subspaces_v4 import orthonormalize
from .fit_distributed_subspaces_v7 import (
    _residualize_fit,
    _ridge_direction,
    augment_bidirectional,
    grouped_oof_ridge,
)
from .resources import require_commit_headroom


RANKS = (2, 4, 8)
HEAD_RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
HEAD_WINDOWS = TOKEN_WINDOWS


def orthonormal_basis(raw_basis: Any) -> Any:
    import torch

    if raw_basis.ndim != 2 or raw_basis.shape[1] == 0:
        raise ValueError("DAS basis must be a nonempty matrix")
    basis, _ = torch.linalg.qr(raw_basis.float(), mode="reduced")
    return basis


def interchange_update(target: Any, source: Any, basis: Any) -> Any:
    if target.shape != source.shape or target.ndim != 2:
        raise ValueError("source and target token activations must align")
    return ((source.float() - target.float()) @ basis @ basis.T).to(target.dtype)


def _encoded_response(
    tokenizer: Any, model: Any, prompt: str, response: str, token_limit: int = 64
) -> tuple[Any, Any, int, int]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    take = min(token_limit, int(response_ids.shape[1]))
    combined = torch.cat((prompt_ids, response_ids[:, :take]), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    return combined, torch.ones_like(combined), int(prompt_ids.shape[1]), take


def capture_layers(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    response_start: int,
    response_tokens: int,
    layer_indices: tuple[int, ...],
) -> dict[int, Any]:
    import torch

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[
                0, response_start : response_start + response_tokens
            ].detach().float()
            return output
        return hook

    layers = _layers(model)
    for index in layer_indices:
        handles.append(layers[index].register_forward_hook(make_hook(index)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_indices):
        raise RuntimeError("DAS capture missed a requested layer")
    return captured


def pooled_window(hidden: Any, window: tuple[int, int]) -> Any:
    start, stop = window
    take_stop = min(stop, hidden.shape[0])
    if start >= take_stop:
        raise ValueError(f"response does not cover token window {window}")
    return hidden[start:take_stop].float().mean(dim=0)


def temporal_head_representation(hidden: Any) -> Any:
    import torch

    width = int(hidden.shape[1])
    values = []
    for start, stop in HEAD_WINDOWS:
        take_stop = min(stop, hidden.shape[0])
        if start < take_stop:
            values.append(hidden[start:take_stop].float().mean(dim=0))
        else:
            values.append(torch.zeros(width, dtype=torch.float32, device=hidden.device))
    return torch.cat(values, dim=0)


def patched_temporal_representation(states: dict[str, Any]) -> Any:
    import torch

    return torch.cat([states[f"head_window_{index}"] for index in range(len(HEAD_WINDOWS))])


def patched_final_state(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    response_start: int,
    response_tokens: int,
    patch_layer: int,
    final_layer: int,
    patch_window: tuple[int, int],
    score_window: tuple[int, int],
    source_hidden: Any,
    raw_basis: Any,
) -> tuple[Any, dict[str, float]]:
    captured: dict[str, Any] = {}
    patch_state = {"tokens": 0, "update_l2": 0.0}
    start, stop = patch_window

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        take_stop = min(stop, response_tokens, source_hidden.shape[0])
        if start >= take_stop:
            raise ValueError("DAS patch window is unavailable")
        target = hidden[0, response_start + start : response_start + take_stop]
        source = source_hidden[start:take_stop].to(target.device)
        basis = orthonormal_basis(raw_basis)
        update = interchange_update(target, source, basis)
        changed = hidden.clone()
        changed[0, response_start + start : response_start + take_stop] += update
        patch_state["tokens"] = int(take_stop - start)
        patch_state["update_l2"] = float(update.detach().float().norm().cpu())
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def final_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        response = hidden[0, response_start : response_start + response_tokens]
        captured["state"] = pooled_window(response, score_window)
        return output

    layers = _layers(model)
    patch_handle = layers[patch_layer].register_forward_hook(patch_hook)
    final_handle = layers[final_layer].register_forward_hook(final_hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        patch_handle.remove()
        final_handle.remove()
    if "state" not in captured:
        raise RuntimeError("DAS patched final state was not captured")
    return captured["state"], patch_state


def patched_final_windows(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    response_start: int,
    response_tokens: int,
    patch_layer: int,
    final_layer: int,
    patch_window: tuple[int, int],
    score_windows: dict[str, tuple[int, int]],
    source_hidden: Any,
    raw_basis: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}
    patch_state: dict[str, Any] = {"tokens": 0, "update_l2": None, "delta_l2": None}
    start, stop = patch_window

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        take_stop = min(stop, response_tokens, source_hidden.shape[0])
        if start >= take_stop:
            raise ValueError("DAS patch window is unavailable")
        target = hidden[0, response_start + start : response_start + take_stop]
        source = source_hidden[start:take_stop].to(target.device)
        update = interchange_update(target, source, orthonormal_basis(raw_basis))
        changed = hidden.clone()
        changed[0, response_start + start : response_start + take_stop] += update
        patch_state["tokens"] = int(take_stop - start)
        patch_state["update_l2_tensor"] = update.float().norm()
        patch_state["delta_l2_tensor"] = (source.float() - target.float()).norm()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def final_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        response = hidden[0, response_start : response_start + response_tokens]
        for name, window in score_windows.items():
            captured[name] = pooled_window(response, window)
        return output

    layers = _layers(model)
    handles = (
        layers[patch_layer].register_forward_hook(patch_hook),
        layers[final_layer].register_forward_hook(final_hook),
    )
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(score_windows):
        raise RuntimeError("DAS final-window capture is incomplete")
    patch_state["update_l2"] = float(patch_state["update_l2_tensor"].detach().cpu())
    patch_state["delta_l2"] = float(patch_state["delta_l2_tensor"].detach().cpu())
    return captured, patch_state


def deterministic_case_split(cases: list[str], axis: str) -> tuple[set[str], set[str]]:
    unique = sorted(set(cases))
    if len(unique) < 5:
        raise ValueError("causal DAS requires at least five fit cases")
    ordered = sorted(
        unique,
        key=lambda case: sha256(f"causal-das-v9:{axis}:{case}".encode()).hexdigest(),
    )
    validation_count = max(2, int(round(0.2 * len(ordered))))
    validation = set(ordered[:validation_count])
    return set(ordered[validation_count:]), validation


def _axis_fit_pairs(
    discovery: dict[str, Any], fit_manifest: dict[str, Any], axis: str
) -> list[dict[str, Any]]:
    allowed = set(fit_manifest["axes"][axis]["fit_case_ids"])
    return [
        pair for pair in discovery["pairs"][axis]
        if pair["split"] == "train" and str(pair["case_id"]) in allowed
    ]


def fit_mean_difference_head(
    cache: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    window: tuple[int, int],
) -> Any:
    import torch

    differences = []
    for pair in pairs:
        low = pooled_window(cache[str(pair["low_record_id"])]["final"], window)
        high = pooled_window(cache[str(pair["high_record_id"])]["final"], window)
        differences.append(high - low)
    head = torch.stack(differences).mean(dim=0).float()
    norm = float(head.norm())
    if norm <= 1e-8:
        raise ValueError("final-layer behavior head has zero norm")
    return (head / head.norm()).detach()


def fit_grouped_ridge_head(
    cache: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    _window: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    import torch

    x = np.stack([
        (
            temporal_head_representation(cache[str(pair["high_record_id"])]["final"])
            - temporal_head_representation(cache[str(pair["low_record_id"])]["final"])
        ).numpy()
        for pair in pairs
    ]).astype(np.float64)
    pair_target = np.ones(len(pairs), dtype=np.float64)
    confounds = np.asarray([
        [float(pair["response_token_delta"])] for pair in pairs
    ], dtype=np.float64)
    groups = np.asarray([str(pair["case_id"]) for pair in pairs])
    design, target, confounds, groups = augment_bidirectional(
        x, pair_target, confounds, groups
    )
    candidates = []
    for alpha in HEAD_RIDGE_ALPHAS:
        result = grouped_oof_ridge(
            design, target, confounds, groups, alpha=alpha,
            folds=min(5, len(np.unique(groups))),
        )
        candidates.append({
            "alpha": alpha, "oof_auroc": result["auroc"],
            "oof_sign_accuracy": result["sign_accuracy"],
        })
    selected = max(
        candidates,
        key=lambda row: (row["oof_auroc"], row["oof_sign_accuracy"], -row["alpha"]),
    )
    x_residual, y_residual, _, _ = _residualize_fit(
        design, target, confounds, design[:1], target[:1], confounds[:1]
    )
    scale = np.std(x_residual, axis=0)
    scale[scale < 1e-6] = 1.0
    scaled = _ridge_direction(
        x_residual / scale, y_residual, float(selected["alpha"])
    )
    raw = scaled / scale
    norm = float(np.linalg.norm(raw))
    if norm <= 1e-10:
        raise ValueError("grouped ridge behavior head has zero norm")
    head = torch.as_tensor((raw / norm).astype(np.float32))
    return head, {"selected": selected, "candidates": candidates}


def evaluate_head(
    head: Any,
    cache: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    _window: tuple[int, int],
) -> dict[str, float]:
    margins = []
    for pair in pairs:
        low = temporal_head_representation(cache[str(pair["low_record_id"])]["final"])
        high = temporal_head_representation(cache[str(pair["high_record_id"])]["final"])
        margins.append(float((high - low).float() @ head.float()))
    return {
        "pairs": len(margins),
        "sign_accuracy": float(np.mean(np.asarray(margins) > 0)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def _prepare_cache(
    model: Any,
    tokenizer: Any,
    texts: dict[str, dict[str, str]],
    record_ids: list[str],
    patch_layer: int,
    final_layer: int,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for index, record_id in enumerate(record_ids):
        row = texts[record_id]
        ids, mask, start, take = _encoded_response(
            tokenizer, model, row["prompt"], row["response"], token_limit=64
        )
        values = capture_layers(
            model, ids, mask, start, take, (patch_layer, final_layer)
        )
        cache[record_id] = {
            "ids": ids.cpu(), "mask": mask.cpu(), "start": start, "take": take,
            "patch": values[patch_layer].cpu(), "final": values[final_layer].cpu(),
        }
        print(f"causal-das-cache {index + 1}/{len(record_ids)}", flush=True)
    return cache


def _initial_basis(
    cache: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    window: tuple[int, int],
    rank: int,
    device: Any,
) -> Any:
    import torch

    start, stop = window
    differences = []
    for pair in pairs:
        low = cache[str(pair["low_record_id"])]["patch"]
        high = cache[str(pair["high_record_id"])]["patch"]
        take_stop = min(stop, low.shape[0], high.shape[0])
        if start < take_stop:
            differences.append(high[start:take_stop] - low[start:take_stop])
    matrix = torch.cat(differences, dim=0).float().to(device)
    _left, _singular, right = torch.linalg.svd(matrix, full_matrices=False)
    if right.shape[0] < rank:
        raise ValueError("not enough activation rank for requested DAS basis")
    return torch.nn.Parameter(right[:rank].T.contiguous())


def _score(state: Any, head: Any) -> Any:
    return state.float() @ head.to(state.device).float()


def train_basis(
    model: Any,
    cache: dict[str, dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    axis: str,
    rank: int,
    patch_layer: int,
    final_layer: int,
    patch_window: tuple[int, int],
    score_windows: dict[str, tuple[int, int]],
    heads: dict[str, Any],
    steps: int,
    learning_rate: float,
    selectivity_weight: float,
    energy_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    raw_basis = _initial_basis(cache, train_pairs, patch_window, rank, torch.device("cuda"))
    optimizer = torch.optim.Adam([raw_basis], lr=learning_rate)
    rng = np.random.default_rng(seed)
    head_score_windows = {
        f"head_window_{index}": window for index, window in enumerate(HEAD_WINDOWS)
    }
    losses: list[float] = []
    target_losses: list[float] = []
    selectivity_losses: list[float] = []
    energy_penalties: list[float] = []
    for step in range(steps):
        pair = train_pairs[int(rng.integers(0, len(train_pairs)))]
        high_to_low = bool(step % 2 == 0)
        source_id = str(pair["high_record_id"] if high_to_low else pair["low_record_id"])
        target_id = str(pair["low_record_id"] if high_to_low else pair["high_record_id"])
        source, target = cache[source_id], cache[target_id]
        ids = target["ids"].to("cuda")
        mask = target["mask"].to("cuda")
        patched, patch_state = patched_final_windows(
            model, ids, mask, int(target["start"]), int(target["take"]),
            patch_layer, final_layer, patch_window, head_score_windows,
            source["patch"], raw_basis,
        )
        patched_representation = patched_temporal_representation(patched)
        target_score = _score(patched_representation, heads[axis])
        source_score = _score(
            temporal_head_representation(source["final"]).to("cuda"), heads[axis]
        ).detach()
        original_score = _score(
            temporal_head_representation(target["final"]).to("cuda"), heads[axis]
        ).detach()
        scale = (source_score - original_score).abs().clamp_min(1.0)
        target_loss = ((target_score - source_score) / scale) ** 2
        selectivity_loss = torch.zeros((), device="cuda")
        for other_axis in heads:
            if other_axis == axis:
                continue
            changed = _score(patched_representation, heads[other_axis])
            original = _score(
                temporal_head_representation(target["final"]).to("cuda"),
                heads[other_axis],
            ).detach()
            selectivity_loss = selectivity_loss + ((changed - original) / scale) ** 2
        energy_ratio = patch_state["update_l2_tensor"] / patch_state[
            "delta_l2_tensor"
        ].clamp_min(1e-6)
        energy_penalty = energy_ratio.square()
        loss = target_loss + selectivity_weight * selectivity_loss + energy_weight * energy_penalty
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_basis], 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        target_losses.append(float(target_loss.detach().cpu()))
        selectivity_losses.append(float(selectivity_loss.detach().cpu()))
        energy_penalties.append(float(energy_penalty.detach().cpu()))
        if (step + 1) % 10 == 0:
            print(f"causal-das-train {axis}/rank-{rank} {step + 1}/{steps} loss={losses[-1]:.4f}", flush=True)
    basis = orthonormal_basis(raw_basis).detach().cpu().numpy().astype(np.float32)
    return basis, {
        "steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
        "mean_last_10_loss": float(np.mean(losses[-10:])),
        "mean_last_10_target_loss": float(np.mean(target_losses[-10:])),
        "mean_last_10_selectivity_loss": float(np.mean(selectivity_losses[-10:])),
        "mean_last_10_energy_penalty": float(np.mean(energy_penalties[-10:])),
    }


def evaluate_basis(
    model: Any,
    cache: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    axis: str,
    basis: np.ndarray,
    patch_layer: int,
    final_layer: int,
    patch_window: tuple[int, int],
    score_windows: dict[str, tuple[int, int]],
    heads: dict[str, Any],
) -> dict[str, Any]:
    import torch

    raw_basis = torch.nn.Parameter(torch.as_tensor(basis, device="cuda"), requires_grad=False)
    head_score_windows = {
        f"head_window_{index}": window for index, window in enumerate(HEAD_WINDOWS)
    }
    improvements: list[float] = []
    direction_success: list[bool] = []
    cross_changes: list[float] = []
    target_changes: list[float] = []
    for pair in pairs:
        for source_key, target_key in (("high_record_id", "low_record_id"), ("low_record_id", "high_record_id")):
            source, target = cache[str(pair[source_key])], cache[str(pair[target_key])]
            with torch.no_grad():
                patched, _ = patched_final_windows(
                    model, target["ids"].to("cuda"), target["mask"].to("cuda"),
                    int(target["start"]), int(target["take"]), patch_layer, final_layer,
                    patch_window, head_score_windows, source["patch"], raw_basis,
                )
                patched_representation = patched_temporal_representation(patched)
                original_target = _score(temporal_head_representation(target["final"]).to("cuda"), heads[axis])
                desired = _score(temporal_head_representation(source["final"]).to("cuda"), heads[axis])
                changed = _score(patched_representation, heads[axis])
                before = float(abs(original_target - desired).cpu())
                after = float(abs(changed - desired).cpu())
                improvements.append((before - after) / max(before, 1.0))
                target_change = float(abs(changed - original_target).cpu())
                target_changes.append(target_change)
                direction_success.append(after < before)
                other = []
                for other_axis in heads:
                    if other_axis == axis:
                        continue
                    original = _score(temporal_head_representation(target["final"]).to("cuda"), heads[other_axis])
                    other.append(float(abs(_score(patched_representation, heads[other_axis]) - original).cpu()))
                cross_changes.append(max(other) if other else 0.0)
    mean_target = float(np.mean(target_changes))
    return {
        "directions": len(improvements),
        "mean_normalized_distance_improvement": float(np.mean(improvements)),
        "direction_success_rate": float(np.mean(direction_success)),
        "mean_target_score_change": mean_target,
        "mean_max_other_axis_change": float(np.mean(cross_changes)),
        "cross_axis_ratio": float(np.mean(cross_changes)) / max(mean_target, 1e-8),
    }


def train_causal_das(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit_manifest = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    subspaces = json.loads(args.subspace_manifest.read_text(encoding="utf-8"))
    texts = _reconstruct(args.packet, args.mapping)
    axes = list(args.axes)
    selected = {
        axis: subspaces["axes"][axis]["selected_on_fit_only"] for axis in axes
    }
    patch_layers = {int(value["layer"]) for value in selected.values()}
    if len(patch_layers) != 1:
        raise ValueError("V9 currently requires axes to share one patch layer")
    patch_layer = next(iter(patch_layers))
    final_layer = 35
    score_windows = {
        axis: tuple(int(value) for value in selected[axis]["window"])
        for axis in axes
    }
    all_pairs = {
        axis: _axis_fit_pairs(discovery, fit_manifest, axis) for axis in axes
    }
    split_pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for axis, pairs in all_pairs.items():
        train_cases, validation_cases = deterministic_case_split(
            [str(pair["case_id"]) for pair in pairs], axis
        )
        split_pairs[axis] = {
            "train": [pair for pair in pairs if str(pair["case_id"]) in train_cases],
            "validation": [
                pair for pair in pairs if str(pair["case_id"]) in validation_cases
            ],
        }
    record_ids = sorted({
        str(pair[key])
        for pairs in all_pairs.values()
        for pair in pairs
        for key in ("low_record_id", "high_record_id")
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = _prepare_cache(
        model, tokenizer, texts, record_ids, patch_layer, final_layer
    )
    heads: dict[str, Any] = {}
    head_results: dict[str, Any] = {}
    for axis in axes:
        head, head_fit = fit_grouped_ridge_head(
            cache, split_pairs[axis]["train"], score_windows[axis]
        )
        heads[axis] = head
        train_result = evaluate_head(
            head, cache, split_pairs[axis]["train"], score_windows[axis]
        )
        validation_result = evaluate_head(
            head, cache, split_pairs[axis]["validation"], score_windows[axis]
        )
        head_results[axis] = {
            "inner_train": train_result,
            "inner_validation": validation_result,
            "fit": head_fit,
            "passes_validation_gate": bool(
                validation_result["sign_accuracy"] >= 0.6
                and validation_result["mean_margin"] > 0
            ),
            "train_case_ids": sorted({
                str(pair["case_id"]) for pair in split_pairs[axis]["train"]
            }),
            "validation_case_ids": sorted({
                str(pair["case_id"]) for pair in split_pairs[axis]["validation"]
            }),
        }
    eligible_axes = [
        axis for axis in axes if head_results[axis]["passes_validation_gate"]
    ]
    rng = np.random.default_rng(2026081001)
    axis_results: dict[str, Any] = {}
    selected_bases: dict[str, np.ndarray] = {}
    for axis_index, axis in enumerate(axes):
        if axis not in eligible_axes:
            axis_results[axis] = {
                "status": "blocked_weak_final_layer_head",
                "head": head_results[axis],
                "rank_candidates": [],
            }
            continue
        rank_candidates: list[dict[str, Any]] = []
        learned_by_rank: dict[int, np.ndarray] = {}
        for rank in args.ranks:
            basis, training = train_basis(
                model, cache, split_pairs[axis]["train"], axis, int(rank),
                patch_layer, final_layer, score_windows[axis], score_windows,
                heads, args.steps, args.learning_rate, args.selectivity_weight,
                args.energy_weight, 2026081100 + axis_index * 10 + int(rank),
            )
            learned = evaluate_basis(
                model, cache, split_pairs[axis]["validation"], axis, basis,
                patch_layer, final_layer, score_windows[axis], score_windows, heads,
            )
            random_basis = orthonormalize(
                rng.standard_normal((basis.shape[0], basis.shape[1]))
            )
            random_result = evaluate_basis(
                model, cache, split_pairs[axis]["validation"], axis, random_basis,
                patch_layer, final_layer, score_windows[axis], score_windows, heads,
            )
            gate = bool(
                learned["mean_normalized_distance_improvement"] > 0
                and learned["mean_normalized_distance_improvement"]
                > random_result["mean_normalized_distance_improvement"] + 0.05
                and learned["direction_success_rate"] >= 0.6
                and learned["cross_axis_ratio"] <= 0.5
            )
            learned_by_rank[int(rank)] = basis
            rank_candidates.append({
                "rank": int(rank), "training": training,
                "inner_validation": learned, "random_control": random_result,
                "passes_surrogate_causal_gate": gate,
                "selection_score": (
                    learned["mean_normalized_distance_improvement"]
                    - random_result["mean_normalized_distance_improvement"]
                    - max(0.0, learned["cross_axis_ratio"] - 0.5)
                ),
            })
        passing = [row for row in rank_candidates if row["passes_surrogate_causal_gate"]]
        if not passing:
            axis_results[axis] = {
                "status": "blocked_no_rank_passed_surrogate_causal_gate",
                "head": head_results[axis], "rank_candidates": rank_candidates,
            }
            continue
        chosen = max(passing, key=lambda row: (row["selection_score"], -row["rank"]))
        rank = int(chosen["rank"])
        full_head, full_head_fit = fit_grouped_ridge_head(
            cache, all_pairs[axis], score_windows[axis]
        )
        full_heads = dict(heads)
        full_heads[axis] = full_head
        final_basis, final_training = train_basis(
            model, cache, all_pairs[axis], axis, rank, patch_layer, final_layer,
            score_windows[axis], score_windows, full_heads, args.final_steps,
            args.learning_rate, args.selectivity_weight, args.energy_weight,
            2026081200 + axis_index,
        )
        random_basis = orthonormalize(
            rng.standard_normal((final_basis.shape[0], final_basis.shape[1]))
        )
        selected_bases[axis] = final_basis
        selected_bases[f"{axis}__random"] = random_basis
        axis_results[axis] = {
            "status": "ready_for_generation_intervention",
            "head": head_results[axis], "rank_candidates": rank_candidates,
            "selected_rank": rank, "final_fit": final_training,
            "full_head_fit": full_head_fit,
            "patch_layer": patch_layer, "patch_window": list(score_windows[axis]),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    if selected_bases:
        np.savez_compressed(args.output / "causal_das_bases.npz", **selected_bases)
    manifest = {
        "type": "downstream_causal_das_fit_v9",
        "status": (
            "ready_for_generation_intervention" if selected_bases
            else "blocked_no_surrogate_causal_subspace"
        ),
        "selection_split": "train_fit_cases_inner_group_holdout",
        "axes_requested": axes,
        "axes_ready": sorted(key for key in selected_bases if "__" not in key),
        "ranks": list(args.ranks), "patch_layer": patch_layer,
        "head_ridge_alphas": list(HEAD_RIDGE_ALPHAS),
        "final_layer": final_layer, "score_windows": {
            key: list(value) for key, value in score_windows.items()
        },
        "head_representation_windows": [list(value) for value in HEAD_WINDOWS],
        "loss": (
            "bidirectional downstream final-state interchange distance plus other-axis "
            "selectivity and differentiable projection-energy penalties"
        ),
        "development_accessed": False, "formal_evaluation_accessed": False,
        "axes": axis_results,
        "evidence_boundary": (
            "A passed surrogate gate only permits blinded generation intervention. "
            "It is not behavioral or attractor evidence."
        ),
    }
    (args.output / "fit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _first_fit_pair(
    discovery: dict[str, Any], fit_manifest: dict[str, Any], axis: str
) -> dict[str, Any]:
    allowed_cases = set(fit_manifest["axes"][axis]["fit_case_ids"])
    return next(
        pair for pair in discovery["pairs"][axis]
        if pair["split"] == "train" and str(pair["case_id"]) in allowed_cases
    )


def feasibility_check(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit_manifest = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    subspaces = json.loads(args.subspace_manifest.read_text(encoding="utf-8"))
    texts = _reconstruct(args.packet, args.mapping)
    axis = args.axis
    selected = subspaces["axes"][axis]["selected_on_fit_only"]
    patch_layer = int(selected["layer"])
    patch_window = tuple(int(value) for value in selected["window"])
    final_layer = 35
    pair = _first_fit_pair(discovery, fit_manifest, axis)
    low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoded: dict[str, tuple[Any, Any, int, int]] = {}
    captured: dict[str, dict[int, Any]] = {}
    for record_id in (low_id, high_id):
        row = texts[record_id]
        encoded[record_id] = _encoded_response(
            tokenizer, model, row["prompt"], row["response"], token_limit=64
        )
        ids, mask, start, take = encoded[record_id]
        captured[record_id] = capture_layers(
            model, ids, mask, start, take, (patch_layer, final_layer)
        )
    low_final = pooled_window(captured[low_id][final_layer], patch_window)
    high_final = pooled_window(captured[high_id][final_layer], patch_window)
    head = high_final - low_final
    head_norm = float(head.norm().cpu())
    if head_norm <= 1e-8:
        raise ValueError("feasibility pair has no final-layer contrast")
    head = (head / head.norm()).detach()
    delta = (
        captured[high_id][patch_layer][patch_window[0] : patch_window[1]]
        - captured[low_id][patch_layer][patch_window[0] : patch_window[1]]
    )
    _left, _singular, right = torch.linalg.svd(delta, full_matrices=False)
    raw_basis = torch.nn.Parameter(right[: args.rank].T.contiguous().to("cuda"))
    ids, mask, start, take = encoded[low_id]
    desired_score = torch.dot(high_final.to("cuda"), head.to("cuda")).detach()
    before_score = torch.dot(low_final.to("cuda"), head.to("cuda")).detach()
    patched, patch_state = patched_final_state(
        model, ids, mask, start, take, patch_layer, final_layer,
        patch_window, patch_window, captured[high_id][patch_layer], raw_basis,
    )
    patched_score = torch.dot(patched.float(), head.to(patched.device).float())
    loss = (patched_score - desired_score) ** 2
    loss.backward()
    gradient_norm = float(raw_basis.grad.detach().float().norm().cpu())
    result = {
        "type": "causal_das_gradient_feasibility_v9",
        "status": "passed" if np.isfinite(gradient_norm) and gradient_norm > 1e-10 else "failed",
        "axis": axis, "pair_id": str(pair["pair_id"]), "rank": args.rank,
        "patch_layer": patch_layer, "final_layer": final_layer,
        "patch_window": list(patch_window), "patch_tokens": patch_state["tokens"],
        "patch_update_l2": patch_state["update_l2"],
        "before_distance": float(abs(before_score - desired_score).cpu()),
        "patched_distance": float(abs(patched_score.detach() - desired_score).cpu()),
        "loss": float(loss.detach().cpu()), "basis_gradient_norm": gradient_norm,
        "model_parameters_trainable": 0,
        "development_accessed": False, "formal_evaluation_accessed": False,
        "evidence_boundary": "Gradient-path feasibility only; not a fitted or evaluated DAS result.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "feasibility.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--subspace-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("specificity", "independence", "non_template"), default="specificity")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--axes", nargs="+", default=["specificity", "independence", "non_template"])
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--final-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--selectivity-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.feasibility_only == args.train:
        raise ValueError("choose exactly one of --feasibility-only or --train")
    result = feasibility_check(args) if args.feasibility_only else train_causal_das(args)
    print(json.dumps({
        "status": result["status"],
        "axes_ready": result.get("axes_ready", []),
    }, indent=2))


if __name__ == "__main__":
    main()
