from __future__ import annotations

from contextlib import nullcontext
import re
import time
from typing import Any, Callable

import numpy as np

from ..directions.hooks import capture_residuals
from ..qualification.parser import parse_yes_no
from .conditions import steer_first_batch_row
from .diagnostics import mean_token_level_kl


OPTION_FINAL_RE = re.compile(r"(?im)^\s*final\s+answer\s*:\s*([A-J])\s*[.!]?\s*$")
OPTION_LINE_RE = re.compile(r"(?i)^\s*(?:final\s+answer\s*:\s*)?([A-J])\s*[.!]?\s*$")


def parse_option_letter(text: str) -> dict[str, Any]:
    final = [value.upper() for value in OPTION_FINAL_RE.findall(text)]
    if final:
        labels = set(final)
        return {
            "parsed_answer": final[-1] if len(labels) == 1 else None,
            "parse_ok": len(labels) == 1,
            "parse_rule": "final_answer_marker" if len(labels) == 1 else "conflicting_final_markers",
        }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = OPTION_LINE_RE.fullmatch(lines[-1]) if lines else None
    if match:
        return {"parsed_answer": match.group(1).upper(), "parse_ok": True, "parse_rule": "last_nonempty_line"}
    return {"parsed_answer": None, "parse_ok": False, "parse_rule": "missing_answer"}


def _encoded_prompt(tokenizer: Any, rendered: str, maximum_input_tokens: int) -> dict[str, Any]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > maximum_input_tokens:
        raise ValueError(f"sealed input exceeds the frozen token limit: {input_tokens}")
    return encoded


def _device_encoded(model: Any, encoded: dict[str, Any]) -> dict[str, Any]:
    device = model.get_input_embeddings().weight.device
    return {key: value.to(device) for key, value in encoded.items()}


def generate_unsteered(
    model: Any,
    tokenizer: Any,
    rendered: str,
    maximum_input_tokens: int,
    maximum_new_tokens: int,
    capture_layer: int,
    parser: Callable[[str], dict[str, Any]] = parse_yes_no,
) -> dict[str, Any]:
    import torch

    encoded = _device_encoded(model, _encoded_prompt(tokenizer, rendered, maximum_input_tokens))
    input_tokens = int(encoded["input_ids"].shape[1])
    current_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    past_key_values = None
    generated_ids: list[int] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), capture_residuals(
        model, [capture_layer], input_tokens - 1,
    ) as captured:
        for _ in range(maximum_new_tokens):
            output = model(
                input_ids=current_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True,
            )
            next_token = int(torch.argmax(output.logits[0, -1]).item())
            generated_ids.append(next_token)
            past_key_values = output.past_key_values
            current_ids = torch.full(
                (1, 1), next_token, device=output.logits.device,
                dtype=encoded["input_ids"].dtype,
            )
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            ], dim=1)
            if next_token == tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_text": text,
        "generated_token_ids": generated_ids,
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "latency_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "token_level_kl": None,
        "activation": captured[capture_layer].numpy(),
        "hook_calls": 0,
        "hook_finite": True,
        **parser(text),
    }


def generate_paired_steered(
    model: Any,
    tokenizer: Any,
    rendered: str,
    maximum_input_tokens: int,
    maximum_new_tokens: int,
    layer: int,
    vector: Any,
    alpha: float,
    parser: Callable[[str], dict[str, Any]] = parse_yes_no,
) -> dict[str, Any]:
    import torch

    encoded = _device_encoded(model, _encoded_prompt(tokenizer, rendered, maximum_input_tokens))
    input_tokens = int(encoded["input_ids"].shape[1])
    current_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    steered_past = None
    reference_past = None
    generated_ids: list[int] = []
    kl_values: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), steer_first_batch_row(model, layer, vector, alpha) as hook_state:
        for _ in range(maximum_new_tokens):
            hook_state["enabled"] = True
            steered_output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=steered_past,
                use_cache=True,
            )
            hook_state["enabled"] = False
            reference_output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=reference_past,
                use_cache=True,
            )
            steered_logits = steered_output.logits[0, -1].double()
            reference_logits = reference_output.logits[0, -1].double()
            steered_log_probability = torch.log_softmax(steered_logits, dim=-1)
            reference_log_probability = torch.log_softmax(reference_logits, dim=-1)
            probability = steered_log_probability.exp()
            kl_values.append(max(0.0, float(torch.sum(
                probability * (steered_log_probability - reference_log_probability),
            ).item())))
            next_token = int(torch.argmax(steered_logits).item())
            generated_ids.append(next_token)
            steered_past = steered_output.past_key_values
            reference_past = reference_output.past_key_values
            current_ids = torch.full(
                (1, 1), next_token, device=steered_logits.device,
                dtype=encoded["input_ids"].dtype,
            )
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            ], dim=1)
            if next_token == tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if hook_state["calls"] < 1 or not hook_state["finite"] or hook_state["initial_activation"] is None:
        raise RuntimeError("sealed paired steering hook failed")
    return {
        "generated_text": text,
        "generated_token_ids": generated_ids,
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "latency_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "token_level_kl": mean_token_level_kl(kl_values),
        "activation": hook_state["initial_activation"].numpy(),
        "hook_calls": int(hook_state["calls"]),
        "hook_finite": bool(hook_state["finite"]),
        **parser(text),
    }


def completion_mean_log_probability(
    model: Any,
    tokenizer: Any,
    rendered: str,
    completion: str,
    maximum_input_tokens: int,
    layer: int | None,
    vector: Any | None,
    alpha: float | None,
) -> float:
    import torch

    encoded = _device_encoded(model, _encoded_prompt(tokenizer, rendered, maximum_input_tokens))
    completion_ids = tokenizer(
        completion, return_tensors="pt", add_special_tokens=False,
    )["input_ids"].to(encoded["input_ids"].device)
    if int(completion_ids.shape[1]) < 1:
        raise ValueError("sealed canonical completion is empty")
    paired = layer is not None and vector is not None and alpha is not None
    current_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    past_key_values = None
    scores: list[float] = []
    context = (
        steer_first_batch_row(model, int(layer), vector, float(alpha)) if paired else nullcontext(None)
    )
    with torch.inference_mode(), context:
        for target in completion_ids[0]:
            output = model(
                input_ids=current_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True,
            )
            log_probability = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
            scores.append(float(log_probability[int(target)].item()))
            past_key_values = output.past_key_values
            batch = 1
            current_ids = target.reshape(1, 1).repeat(batch, 1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            ], dim=1)
    return float(np.mean(scores))
