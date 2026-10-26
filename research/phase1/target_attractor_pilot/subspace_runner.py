from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .factorial import AXES, factorial_rows
from .subspaces import (
    SubspaceTarget, attract_to_subspaces, fit_subspace_target,
    norm_matched_random_subspace, principal_cosines, target_distance,
)
from research.phase1.template_attractor_pilot.data import build_cases
from research.phase1.template_attractor_pilot.pilot import (
    _layers, capture_layers, generated_metrics, relative_layers, render_prompt,
)


WINDOW_STEPS = {"early_4": 4, "early_8": 8, "early_16": 16}
CONDITIONS = (
    "baseline", "specificity_only", "independence_only", "non_template_only",
    "combined", "rank_and_norm_matched_random",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_or_load_activations(
    model: Any, tokenizer: Any, layer: int, cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    import torch

    rows = factorial_rows()
    index_path = cache_dir / "factorial_rows.json"
    activation_path = cache_dir / "factorial_activations.npz"
    if index_path.is_file() and activation_path.is_file():
        stored_rows = json.loads(index_path.read_text(encoding="utf-8"))
        if [row["record_id"] for row in stored_rows] != [row["record_id"] for row in rows]:
            raise ValueError("cached factorial activation order does not match the frozen design")
        with np.load(activation_path) as archive:
            return stored_rows, {window: archive[window].astype(np.float32) for window in WINDOW_STEPS}
    device = model.get_input_embeddings().weight.device
    values: dict[str, list[np.ndarray]] = {window: [] for window in WINDOW_STEPS}
    for index, row in enumerate(rows):
        prompt = tokenizer(
            render_prompt(tokenizer, str(row["prompt"])), add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        response = tokenizer(str(row["response"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        combined = torch.cat((prompt, response), dim=1).to(device)
        positions = {
            window: int(prompt.shape[1]) + min(step, int(response.shape[1])) - 1
            for window, step in WINDOW_STEPS.items()
        }
        with torch.inference_mode(), capture_layers(model, [layer], positions) as captured:
            model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False)
        for window in WINDOW_STEPS:
            values[window].append(captured[window][layer])
        if (index + 1) % 16 == 0:
            print(f"factorial_activation {index + 1}/{len(rows)}", flush=True)
    arrays = {window: np.stack(items).astype(np.float32) for window, items in values.items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(index_path, rows)
    np.savez_compressed(activation_path, **arrays)
    return rows, arrays


def _contrasts(rows: list[dict[str, Any]], matrix: np.ndarray, axis: str) -> dict[str, np.ndarray]:
    from .factorial import matched_factor_contrasts

    return matched_factor_contrasts(rows, matrix, axis)


def fit_targets(
    rows: list[dict[str, Any]], activations: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, SubspaceTarget]], dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    development_indices = [index for index, row in enumerate(rows) if row["split"] == "development"]
    train_rows = [rows[index] for index in train_indices]
    targets: dict[str, dict[str, SubspaceTarget]] = {}
    diagnostics: dict[str, Any] = {"windows": {}, "overlap": {}}
    for window, matrix in activations.items():
        targets[window] = {}
        diagnostics["windows"][window] = {}
        for axis in AXES:
            contrast = _contrasts(train_rows, matrix[train_indices], axis)
            target = fit_subspace_target(contrast["differences"], contrast["positive"])
            targets[window][axis] = target
            labels = np.asarray([int(rows[index][axis]) for index in development_indices])
            score = -target_distance(matrix[development_indices], target)
            diagnostics["windows"][window][axis] = {
                "rank": target.rank,
                "explained_variance": target.explained_variance,
                "development_auroc": float(roc_auc_score(labels, score)),
            }
        for left_index, left in enumerate(AXES):
            for right in AXES[left_index + 1:]:
                cosines = principal_cosines(targets[window][left].basis, targets[window][right].basis)
                diagnostics["overlap"][f"{window}:{left}:{right}"] = {
                    "maximum_principal_cosine": float(cosines.max()),
                    "mean_principal_cosine": float(cosines.mean()),
                }
    return targets, diagnostics


def _targets_for_condition(
    condition: str,
    targets: dict[str, dict[str, SubspaceTarget]],
    random_targets: dict[str, list[SubspaceTarget]],
) -> dict[str, list[SubspaceTarget]]:
    if condition == "rank_and_norm_matched_random":
        return random_targets
    if condition == "combined":
        axes = AXES
    elif condition.endswith("_only"):
        axes = (condition.removesuffix("_only"),)
    elif condition == "baseline":
        axes = ()
    else:
        raise ValueError(f"unknown condition: {condition}")
    return {window: [targets[window][axis] for axis in axes] for window in WINDOW_STEPS}


def _behavior_metrics(text: str, anchors: tuple[str, ...]) -> dict[str, float]:
    metrics = generated_metrics(text, anchors)
    independence_markers = ("替代解释", "不能直接", "不足以", "排除", "区分", "验证")
    metrics["independence_marker_density"] = (
        sum(text.count(marker) for marker in independence_markers) / max(len(text), 1) * 100.0
    )
    metrics["repetition_ratio"] = 1.0 - len(set(text)) / max(len(text), 1)
    return metrics


def generate_condition(
    model: Any, tokenizer: Any, prompt: str, anchors: tuple[str, ...], layer: int,
    condition_targets: dict[str, list[SubspaceTarget]], condition: str,
    gain: float, maximum_delta_norm: float, maximum_new_tokens: int,
    begin_after_tokens: int,
) -> dict[str, Any]:
    import torch

    encoded = tokenizer(
        render_prompt(tokenizer, prompt), add_special_tokens=False, return_tensors="pt",
    )
    device = model.get_input_embeddings().weight.device
    current_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(current_ids)
    past_key_values = None
    generated: list[int] = []
    with torch.inference_mode(), attract_to_subspaces(
        model, layer, condition_targets, gain, maximum_delta_norm,
    ) as state:
        for step in range(maximum_new_tokens):
            state["enabled"] = bool(condition != "baseline" and step >= begin_after_tokens)
            state["window"] = "early_16" if begin_after_tokens >= 16 else (
                "early_4" if step < 8 else "early_8" if step < 16 else "early_16"
            )
            output = model(
                input_ids=current_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True,
            )
            token = int(torch.argmax(output.logits[0, -1]).item())
            generated.append(token)
            past_key_values = output.past_key_values
            current_ids = torch.tensor([[token]], device=device, dtype=encoded["input_ids"].dtype)
            attention_mask = torch.cat((attention_mask, torch.ones_like(current_ids)), dim=1)
            if token == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("empty subspace-controlled generation")
    return {
        "condition": condition,
        "text": text,
        "output_tokens": len(generated),
        "generation_finished": len(generated) < maximum_new_tokens,
        "hook_calls": int(state["calls"]),
        "mean_delta_norm": float(np.mean(state["delta_norms"])) if state["delta_norms"] else 0.0,
        **_behavior_metrics(text, anchors),
    }


def run(
    checkpoint: Path, output: Path, gain: float = 0.25, maximum_delta_norm: float = 4.0,
    case_limit: int = 2, maximum_new_tokens: int = 384,
    begin_after_tokens: int = 4, cache_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if "evaluation" in str(output).lower():
        raise ValueError("subspace discovery output must not use an evaluation path")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16,
        device_map="auto", quantization_config=quantization,
    )
    model.eval()
    candidate_layers = relative_layers(len(_layers(model)))
    layer = min(candidate_layers, key=lambda value: abs(value - (len(_layers(model)) - 1) * 0.5))
    activation_cache = cache_dir if cache_dir is not None else output / "cache"
    rows, activations = extract_or_load_activations(model, tokenizer, layer, activation_cache)
    targets, diagnostics = fit_targets(rows, activations)
    hidden_size = int(next(iter(targets["early_4"].values())).basis.shape[0])
    random_targets = {
        window: [
            norm_matched_random_subspace(targets[window][axis], hidden_size, 20260726 + index * 101 + WINDOW_STEPS[window])
            for index, axis in enumerate(AXES)
        ]
        for window in WINDOW_STEPS
    }
    cases = [case for case in build_cases() if case.split == "development"][:case_limit]
    records = []
    for case in cases:
        for condition in CONDITIONS:
            record = generate_condition(
                model, tokenizer, case.prompt, case.anchors, layer,
                _targets_for_condition(condition, targets, random_targets), condition,
                gain, maximum_delta_norm, maximum_new_tokens, begin_after_tokens,
            )
            record.update({"case_id": case.case_id, "split": "development", "prompt": case.prompt})
            records.append(record)
            print(f"subspace_generation {case.case_id} {condition}", flush=True)
    summaries = {}
    for condition in CONDITIONS:
        selected = [row for row in records if row["condition"] == condition]
        summaries[condition] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "anchor_coverage", "formulaic_structure_score", "independence_marker_density",
                "characters", "repetition_ratio", "output_tokens", "mean_delta_norm",
            )
        }
        summaries[condition]["completion_rate"] = float(np.mean([row["generation_finished"] for row in selected]))
    result = {
        "experiment": "target-subspaces-causal-smoke-v1",
        "scope": "discovery_only",
        "formal_evaluation_records_accessed": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "layer": layer,
        "gain": gain,
        "maximum_delta_norm": maximum_delta_norm,
        "maximum_new_tokens": maximum_new_tokens,
        "begin_after_tokens": begin_after_tokens,
        "activation_cache": str(activation_cache),
        "factorial_record_count": len(rows),
        "diagnostics": diagnostics,
        "summaries": summaries,
        "records": records,
    }
    _write_json(output / "subspace_smoke_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=0.25)
    parser.add_argument("--maximum-delta-norm", type=float, default=4.0)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=384)
    parser.add_argument("--begin-after-tokens", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(
        args.checkpoint, args.output, args.gain, args.maximum_delta_norm,
        args.case_limit, args.maximum_new_tokens, args.begin_after_tokens, args.cache_dir,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
