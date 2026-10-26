from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt
from research.phase1.target_attractor_pilot.sae_causal.resources import require_commit_headroom

from .control import (
    ClosedLoopConfig,
    ClosedLoopGenerationHook,
    EnsembleLinearObserver,
    LowRankFeedbackPolicy,
    Perturbation,
    recovery_metrics,
)


PROMPT = (
    "Write a concise three-sentence review of a newly opened neighborhood library. "
    "Mention its collection, study space, and staff. Do not use headings or bullet points."
)
POSITIVE_RESPONSE = (
    "The library has a thoughtfully chosen collection with enough depth to reward repeat visits. "
    "Its quiet, well-lit study space makes sustained work comfortable, and the staff offer precise help without hovering. "
    "It already feels like a genuinely useful addition to the neighborhood."
)
NEGATIVE_RESPONSE = (
    "The library's thin and poorly organized collection makes it difficult to find anything worthwhile. "
    "Its cramped, noisy study space discourages concentration, and the staff are dismissive when asked for help. "
    "It currently feels like a disappointing addition to the neighborhood."
)
CONDITIONS = (
    "baseline",
    "perturbed_no_control",
    "perturbed_always_on",
    "perturbed_closed_loop",
    "perturbed_closed_loop_withdrawal",
)


def _encoded_pair(tokenizer: Any, model: Any, prompt: str, response: str) -> tuple[Any, Any, int, int]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    combined = torch.cat((prompt_ids, response_ids), dim=1).to(model.get_input_embeddings().weight.device)
    return combined, torch.ones_like(combined), int(prompt_ids.shape[1]), int(response_ids.shape[1])


def _capture_response_mean(
    model: Any, tokenizer: Any, layer_index: int, prompt: str, response: str
) -> np.ndarray:
    import torch

    input_ids, attention_mask, response_start, response_tokens = _encoded_pair(
        tokenizer, model, prompt, response
    )
    captured: dict[str, np.ndarray] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        response_hidden = hidden[0, response_start : response_start + response_tokens]
        captured["mean"] = response_hidden.detach().float().mean(dim=0).cpu().numpy()
        return output

    handle = _layers(model)[layer_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    if "mean" not in captured:
        raise RuntimeError("sentiment smoke failed to capture donor activations")
    return captured["mean"]


def _sentiment_components(
    positive: np.ndarray, negative: np.ndarray, window_size: int
) -> tuple[EnsembleLinearObserver, LowRankFeedbackPolicy, float, float]:
    direction = np.asarray(positive - negative, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ValueError("sentiment donor direction has zero norm")
    direction /= norm
    hidden_width = len(direction)
    weights = np.zeros((3, 1, hidden_width * 2), dtype=np.float32)
    weights[:, 0, :hidden_width] = direction
    observer = EnsembleLinearObserver(
        weights=weights,
        bias=np.zeros((3, 1), dtype=np.float32),
        feature_mean=np.zeros(hidden_width * 2, dtype=np.float32),
        feature_scale=np.ones(hidden_width * 2, dtype=np.float32),
        window_size=window_size,
        axes=("sentiment",),
    )
    policy = LowRankFeedbackPolicy(
        basis=direction[:, None],
        error_gain=np.ones((1, 1), dtype=np.float32),
        velocity_gain=np.zeros((1, 1), dtype=np.float32),
        axes=("sentiment",),
    )
    positive_anchor = float(positive @ direction)
    negative_anchor = float(negative @ direction)
    return observer, policy, positive_anchor, negative_anchor


def _generate(
    model: Any,
    tokenizer: Any,
    layer_index: int,
    observer: EnsembleLinearObserver,
    policy: LowRankFeedbackPolicy,
    config: ClosedLoopConfig,
    perturbations: list[Perturbation],
    seed: int,
    maximum_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    import torch

    rendered = render_prompt(tokenizer, PROMPT)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.get_input_embeddings().weight.device)
    controller = ClosedLoopGenerationHook(observer, policy, config, perturbations)
    handle = _layers(model)[layer_index].register_forward_hook(controller)
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=maximum_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("sentiment smoke generated an empty response")
    return text, controller.summary()


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "type": "closed_loop_sentiment_positive_control_v1",
        "status": "dry_run_passed" if args.dry_run else "running",
        "conditions": list(CONDITIONS),
        "layer": args.layer,
        "perturbation_step": args.perturbation_step,
        "maximum_new_tokens": args.maximum_new_tokens,
        "formal_evaluation_accessed": False,
        "evidence_boundary": (
            "Engineering positive control only. Internal-state recovery is not behavioral "
            "recovery and is not evidence for an internal attractor."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (args.output / "manifest.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan
    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        memory = require_commit_headroom(args.minimum_commit_headroom_gb)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    except (OSError, RuntimeError) as error:
        blocked = {
            **plan,
            "status": "blocked_model_load",
            "model": str(args.model),
            "error_type": type(error).__name__,
            "error": str(error),
            "next_action": "restore Windows commit headroom and rerun the identical command",
        }
        (args.output / "blocked_runtime.json").write_text(
            json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    layer_count = len(_layers(model))
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"layer {args.layer} is outside model layer count {layer_count}")
    model.eval()
    positive = _capture_response_mean(model, tokenizer, args.layer, PROMPT, POSITIVE_RESPONSE)
    negative = _capture_response_mean(model, tokenizer, args.layer, PROMPT, NEGATIVE_RESPONSE)
    observer, policy, positive_anchor, negative_anchor = _sentiment_components(
        positive, negative, args.observer_window
    )
    anchor_gap = max(abs(positive_anchor - negative_anchor), 1e-3)
    direction = policy.basis[:, 0]
    perturbation = Perturbation(
        step=args.perturbation_step,
        update=-direction * anchor_gap * args.perturbation_multiplier,
        family="reverse_sentiment_direction",
    )
    maximum_step_norm = anchor_gap * args.maximum_step_fraction
    energy_budget = maximum_step_norm**2 * args.maximum_control_steps
    records = []
    for condition in CONDITIONS:
        gate_mode = {
            "baseline": "disabled",
            "perturbed_no_control": "disabled",
            "perturbed_always_on": "always_on",
            "perturbed_closed_loop": "closed_loop",
            "perturbed_closed_loop_withdrawal": "closed_loop",
        }[condition]
        withdrawal = (
            args.perturbation_step + args.withdrawal_after_tokens
            if condition == "perturbed_closed_loop_withdrawal" else None
        )
        config = ClosedLoopConfig(
            axis="sentiment",
            target=positive_anchor,
            deadband=anchor_gap * args.deadband_fraction,
            maximum_uncertainty=1e-6,
            control_interval=args.control_interval,
            start_step=0,
            stop_step=args.maximum_new_tokens,
            maximum_step_norm=maximum_step_norm,
            total_energy_budget=energy_budget,
            gate_mode=gate_mode,
            withdrawal_step=withdrawal,
        )
        applied_perturbations = [] if condition == "baseline" else [perturbation]
        response, state = _generate(
            model, tokenizer, args.layer, observer, policy, config,
            applied_perturbations, args.seed, args.maximum_new_tokens,
        )
        recovery = (
            None if condition == "baseline" else recovery_metrics(
                state["trace"], "sentiment", args.perturbation_step,
                positive_anchor, anchor_gap * args.recovery_tolerance_fraction,
                consecutive_steps=args.recovery_consecutive_steps,
            )
        )
        records.append({
            "condition": condition,
            "seed": args.seed,
            "response": response,
            "controller": state,
            "internal_recovery": recovery,
        })
        print(f"sentiment-positive-control {len(records)}/{len(CONDITIONS)} {condition}", flush=True)
    result = {
        **plan,
        "status": "engineering_smoke_complete",
        "model": str(args.model),
        "windows_commit_memory_before_load": None if memory is None else {
            "committed_gb": memory.committed_gb,
            "limit_gb": memory.limit_gb,
            "headroom_gb": memory.headroom_gb,
        },
        "positive_anchor": positive_anchor,
        "negative_anchor": negative_anchor,
        "anchor_gap": anchor_gap,
        "maximum_step_norm": maximum_step_norm,
        "total_energy_budget": energy_budget,
        "records": records,
        "behavioral_scoring": "pending_blind_scoring",
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026072801)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--observer-window", type=int, default=4)
    parser.add_argument("--control-interval", type=int, default=1)
    parser.add_argument("--perturbation-step", type=int, default=12)
    parser.add_argument("--perturbation-multiplier", type=float, default=1.0)
    parser.add_argument("--maximum-step-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-control-steps", type=int, default=16)
    parser.add_argument("--deadband-fraction", type=float, default=0.1)
    parser.add_argument("--recovery-tolerance-fraction", type=float, default=0.2)
    parser.add_argument("--recovery-consecutive-steps", type=int, default=2)
    parser.add_argument("--withdrawal-after-tokens", type=int, default=8)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "conditions": result["conditions"],
        "formal_evaluation_accessed": result["formal_evaluation_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()
