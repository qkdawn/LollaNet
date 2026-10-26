from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.target_attractor_pilot.core import (
    Schedule, intervention_at_step, norm_matched_random_direction,
    matched_dynamics, scheduled_residual_intervention, summarize_trace,
)
from research.phase1.target_attractor_pilot.data import direction_rows, smoke_cases
from research.phase1.template_attractor_pilot.pilot import (
    _layers, capture_layers, generated_metrics, relative_layers, render_prompt,
)


CONDITIONS = (
    "baseline", "continuous_target", "target_pulse_release",
    "target_pulse_release_perturb", "random_pulse_release",
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fit_target_directions(model: Any, tokenizer: Any, layers: list[int]) -> dict[int, np.ndarray]:
    import torch

    device = model.get_input_embeddings().weight.device
    paired: dict[int, dict[str, dict[int, np.ndarray]]] = {layer: {} for layer in layers}
    for row in direction_rows():
        rendered = render_prompt(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
        response_ids = tokenizer(str(row["response"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        combined = torch.cat([prompt_ids, response_ids], dim=1).to(device)
        position = int(combined.shape[1]) - 1
        with torch.inference_mode(), capture_layers(model, layers, {"end": position}) as captured:
            model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False)
        for layer in layers:
            paired[layer].setdefault(str(row["case_id"]), {})[int(row["label"])] = captured["end"][layer]
    directions = {}
    for layer in layers:
        differences = [values[1] - values[0] for values in paired[layer].values() if set(values) == {0, 1}]
        direction = np.mean(np.stack(differences), axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"target direction is degenerate at layer {layer}")
        directions[layer] = (direction / norm).astype(np.float32)
    return directions


def generate_scheduled(
    model: Any, tokenizer: Any, prompt: str, anchors: tuple[str, ...], layer: int,
    direction: np.ndarray, random_direction: np.ndarray, monitor_layer: int,
    monitor_direction: np.ndarray, condition: str, strength: float,
    schedule: Schedule, maximum_new_tokens: int,
) -> dict[str, Any]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    current_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(current_ids)
    past_key_values = None
    generated: list[int] = []
    with torch.inference_mode(), scheduled_residual_intervention(
        model, layer, direction, random_direction, strength, monitor_layer, monitor_direction,
    ) as hook_state:
        for step in range(maximum_new_tokens):
            hook_state["step"] = step
            hook_state["mode"] = intervention_at_step(condition, step, schedule)
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
    if not hook_state["finite"]:
        raise RuntimeError("non-finite hidden state during scheduled generation")
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("empty scheduled generation")
    return {
        "condition": condition,
        "strength": strength,
        "maximum_new_tokens": maximum_new_tokens,
        "text": text,
        "output_tokens": len(generated),
        "hook_calls": int(hook_state["calls"]),
        "monitor_calls": int(hook_state["monitor_calls"]),
        "trace": hook_state["trace"],
        "trace_summary": summarize_trace(hook_state["trace"], schedule),
        **generated_metrics(text, anchors),
    }


def run(checkpoint: Path, output: Path, strength: float, case_limit: int, maximum_new_tokens: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if "evaluation" in str(output).lower():
        raise ValueError("target-attractor discovery output must not use an evaluation path")
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
    layers = relative_layers(len(_layers(model)))
    layer = min(layers, key=lambda value: abs(value - (len(_layers(model)) - 1) * 0.5))
    monitor_layer = max(value for value in layers if value > layer)
    directions = fit_target_directions(model, tokenizer, [layer, monitor_layer])
    direction = directions[layer]
    monitor_direction = directions[monitor_layer]
    random_direction = norm_matched_random_direction(direction, 20260726)
    schedule = Schedule()
    records = []
    for case in smoke_cases(case_limit):
        for condition in CONDITIONS:
            record = generate_scheduled(
                model, tokenizer, case.prompt, case.anchors, layer, direction,
                random_direction, monitor_layer, monitor_direction, condition, strength,
                schedule, maximum_new_tokens,
            )
            record.update({"case_id": case.case_id, "split": "development", "prompt": case.prompt})
            records.append(record)
            print(f"target_attractor {case.case_id} {condition}", flush=True)
    baseline = {row["case_id"]: row for row in records if row["condition"] == "baseline"}
    persistence = []
    recovery = []
    for row in records:
        base = baseline[row["case_id"]]["trace_summary"]
        summary = row["trace_summary"]
        if row["condition"] in {"target_pulse_release", "target_pulse_release_perturb"}:
            if summary["release_mean"] is not None and base["release_mean"] is not None:
                persistence.append(float(summary["release_mean"] - base["release_mean"]))
        if row["condition"] == "target_pulse_release_perturb":
            if summary["recovery_mean"] is not None and summary["perturbation_mean"] is not None:
                recovery.append(float(summary["recovery_mean"] - summary["perturbation_mean"]))
    result = {
        "experiment": "target-attractor-smoke-v1",
        "scope": "discovery_only",
        "formal_evaluation_records_accessed": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "layer": layer,
        "monitor_layer": monitor_layer,
        "strength": strength,
        "maximum_new_tokens": maximum_new_tokens,
        "schedule": schedule.__dict__,
        "direction_sha256": hashlib.sha256(direction.tobytes()).hexdigest(),
        "random_norm_ratio": float(np.linalg.norm(random_direction) / np.linalg.norm(direction)),
        "random_target_cosine": float(random_direction @ direction / (np.linalg.norm(random_direction) * np.linalg.norm(direction))),
        "mean_release_projection_delta_vs_baseline": float(np.mean(persistence)) if persistence else None,
        "mean_post_perturbation_recovery_delta": float(np.mean(recovery)) if recovery else None,
        "matched_dynamics": matched_dynamics(records),
        "interpretation": "candidate_signal_only; persistence and recovery are required before using attractor language",
        "records": records,
    }
    _json(output / "smoke_results.json", result)
    np.save(output / "target_direction.npy", direction)
    np.save(output / "monitor_direction.npy", monitor_direction)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=80)
    args = parser.parse_args()
    result = run(args.checkpoint, args.output, args.strength, args.case_limit, args.maximum_new_tokens)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
