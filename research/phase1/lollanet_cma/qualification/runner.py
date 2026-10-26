from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import yaml

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    dependency_lock,
    normalized_local_checkpoint,
    read_json,
    read_jsonl,
    runtime_code_identity,
    runtime_environment,
    sha256_file,
    stable_hash,
)
from .hooks import residual_probe, transformer_layers
from .parser import parse_yes_no
from ..access_views.views import validate_access_view


SYSTEM_INSTRUCTION = (
    "Solve the causal reasoning problem using only the stated hypothetical world. "
    "End the response with exactly one line in the form `Final answer: yes` or `Final answer: no`."
)


def _validate_non_thinking_candidate(candidate: dict[str, Any]) -> None:
    model_id = str(candidate.get("model_id", "")).lower()
    forbidden_markers = ("thinking", "deepseek-r1", "r1-distill")
    if candidate.get("reasoning_mode") != "disabled":
        raise ValueError("qualification candidate must explicitly disable reasoning mode")
    if any(marker in model_id for marker in forbidden_markers):
        raise ValueError("thinking/reasoning model is forbidden by the qualification charter")


def _model_placement(model: Any) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        devices = sorted({str(value) for value in device_map.values()})
    else:
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
    input_device = model.get_input_embeddings().weight.device
    disallowed = {"cpu", "disk", "meta"}
    loaded_on_cuda = str(input_device).startswith("cuda") and not any(
        device in disallowed or device.startswith("cpu") for device in devices
    )
    return {"input_device": str(input_device), "devices": devices, "loaded_on_cuda": loaded_on_cuda}


def _render(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_prefix(path: Path, expected_ids: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if len(rows) > len(expected_ids) or observed != expected_ids[:len(rows)] or len(observed) != len(set(observed)):
        raise ValueError(f"{label} partial artifact is not an exact ordered prefix")
    return rows


def _write_progress(path: Path, candidate_id: str, stage: str, completed: int, total: int) -> None:
    atomic_write_json(path, {
        "type": "lollanet_cma_qualification_progress_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "stage": stage,
        "completed": completed,
        "total": total,
    })


def _generate_one(
    model: Any, tokenizer: Any, prompt: str, max_input_tokens: int, max_new_tokens: int,
    layer_index: int | None = None, probe_delta: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import torch
    rendered = _render(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        raise ValueError(f"input exceeds frozen token limit: {input_tokens} > {max_input_tokens}")
    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    torch.cuda.reset_peak_memory_stats()
    context = residual_probe(model, int(layer_index), probe_delta) if layer_index is not None else nullcontext(None)
    started = time.perf_counter()
    with torch.inference_mode(), context as hook_state:
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    continuation = generated[0, input_tokens:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    parsed = parse_yes_no(text)
    row = {
        "generated_text": text,
        **parsed,
        "input_tokens": input_tokens,
        "output_tokens": int(continuation.shape[0]),
        "latency_seconds": latency,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return row, hook_state


def run_candidate(
    charter_path: Path, access_view: Path, readiness_path: Path, candidate_id: str, output_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    readiness = read_json(readiness_path)
    content = dict(readiness)
    claimed = content.pop("readiness_content_sha256", None)
    if claimed != stable_hash(content):
        raise ValueError("qualification readiness content hash mismatch")
    if readiness.get("charter", {}).get("sha256") != sha256_file(charter_path):
        raise ValueError("qualification readiness used a different charter")
    access_manifest_path = access_view / "manifest.json"
    if readiness.get("access_view", {}).get("manifest_sha256") != sha256_file(access_manifest_path):
        raise ValueError("qualification readiness used a different access view")
    if readiness.get("access_view", {}).get("path") != str(access_view.resolve()):
        raise ValueError("qualification access view path differs from readiness")
    if readiness.get("access_view", {}).get("validator_code") != runtime_code_identity(Path(__file__).parents[1] / "access_views"):
        raise ValueError("qualification access-view validator code differs from readiness")
    access_validation = validate_access_view(access_view, "qualification")
    if readiness.get("access_view", {}).get("validation") != access_validation:
        raise ValueError("qualification access view no longer matches readiness")
    if readiness.get("code") != runtime_code_identity(Path(__file__).parent):
        raise ValueError("qualification code differs from the readiness audit")
    if readiness.get("dependency_lock") != dependency_lock():
        raise ValueError("qualification dependency lock differs from the readiness audit")
    if readiness.get("environment") != runtime_environment():
        raise ValueError("qualification environment differs from the readiness audit")
    ready = {row["candidate_id"]: row for row in readiness["candidates"]}
    if candidate_id not in ready or ready[candidate_id]["status"] != "ready":
        raise ValueError(f"candidate is not ready for qualification: {candidate_id}")
    candidate = next(row for row in charter["candidates"] if row["id"] == candidate_id)
    _validate_non_thinking_candidate(candidate)
    checkpoint = normalized_local_checkpoint(charter_path, candidate.get("local_checkpoint"))
    if (
        candidate.get("revision") != ready[candidate_id].get("revision_expected")
        or (str(checkpoint) if checkpoint else None) != ready[candidate_id].get("local_checkpoint")
    ):
        raise ValueError("candidate charter differs from the readiness audit")
    config = charter["loading"]
    if not candidate.get("revision") or checkpoint is None:
        raise ValueError("candidate revision and local checkpoint must be frozen")
    access_manifest = read_json(access_manifest_path)
    access_artifacts = access_manifest["artifacts"]
    prompts = read_jsonl(Path(access_artifacts["prompts"]["path"]))
    controller = {row["record_id"]: row for row in read_jsonl(Path(access_artifacts["controller"]["path"]))}
    evaluator = {row["record_id"]: row for row in read_jsonl(Path(access_artifacts["evaluator"]["path"]))}
    if len(prompts) != int(charter["qualification"]["records"]):
        raise ValueError("Stage 0 qualification split does not match the charter")
    smoke_count = int(charter["qualification"]["hook_smoke_records"])
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = output_dir / "hook_smoke.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    intent_path = output_dir / "run_intent.json"
    intent = {
        "type": "lollanet_cma_qualification_run_intent_v1",
        "candidate_id": candidate_id,
        "model_id": candidate["model_id"],
        "reasoning_mode": candidate["reasoning_mode"],
        "revision": candidate["revision"],
        "local_checkpoint": str(checkpoint),
        "charter_sha256": sha256_file(charter_path),
        "stage0_manifest_sha256": access_validation["source_stage0_manifest_sha256"],
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "readiness_sha256": sha256_file(readiness_path),
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
    }
    intent["intent_content_sha256"] = stable_hash(intent)
    if intent_path.is_file():
        if read_json(intent_path) != intent:
            raise ValueError("existing qualification run intent differs from the current frozen contract")
    else:
        atomic_write_json(intent_path, intent)
    smoke_ids = [str(row["record_id"]) for row in prompts[:smoke_count]]
    prediction_ids = [str(row["record_id"]) for row in prompts]
    smoke = _load_prefix(smoke_path, smoke_ids, "hook smoke")
    predictions = _load_prefix(predictions_path, prediction_ids, "qualification predictions")

    set_seed(int(config["seed"]))
    dtype = torch.bfloat16 if config["compute_dtype"] == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    placement = _model_placement(model)
    if not placement["loaded_on_cuda"]:
        raise RuntimeError(f"candidate was not loaded entirely on CUDA: {placement}")
    layers = transformer_layers(model)
    layer_index = round((len(layers) - 1) * 0.5)
    for prompt_row in prompts[len(smoke):smoke_count]:
        generated, hook = _generate_one(
            model, tokenizer, prompt_row["prompt"], int(config["maximum_input_tokens"]),
            int(config["maximum_new_tokens"]), layer_index, float(config["hook_probe_delta"]),
        )
        if hook is None or hook["calls"] < 1 or not hook["finite"] or not hook["modified"]:
            raise RuntimeError(f"hook smoke failed for {prompt_row['record_id']}: {hook}")
        smoke.append({"record_id": prompt_row["record_id"], **generated, "hook": hook})
        atomic_write_jsonl(smoke_path, smoke)
        _write_progress(progress_path, candidate_id, "hook_smoke", len(smoke), smoke_count)
    for prompt_row in prompts[len(predictions):]:
        generated, _ = _generate_one(
            model, tokenizer, prompt_row["prompt"], int(config["maximum_input_tokens"]),
            int(config["maximum_new_tokens"]),
        )
        record_id = prompt_row["record_id"]
        answer = evaluator[record_id]["answer"]
        predictions.append({
            "record_id": record_id,
            "model_family": controller[record_id]["model_family"],
            "query_type": controller[record_id]["query_type"],
            "gold_answer": answer,
            "correct": bool(generated["parse_ok"] and generated["parsed_answer"] == answer),
            **generated,
        })
        atomic_write_jsonl(predictions_path, predictions)
        _write_progress(progress_path, candidate_id, "qualification", len(predictions), len(prompts))
    family_accuracy = {
        family: float(np.mean([row["correct"] for row in predictions if row["model_family"] == family]))
        for family in sorted({row["model_family"] for row in predictions})
    }
    manifest = {
        "type": "lollanet_cma_qualification_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "model_id": candidate["model_id"],
        "reasoning_mode": candidate["reasoning_mode"],
        "revision": candidate["revision"],
        "local_checkpoint": str(checkpoint.resolve()),
        "quantization": "bitsandbytes_nf4_4bit",
        "compute_dtype": config["compute_dtype"],
        "batch_size": 1,
        "maximum_input_tokens": int(config["maximum_input_tokens"]),
        "maximum_new_tokens": int(config["maximum_new_tokens"]),
        "sampling": "greedy",
        "seed": int(config["seed"]),
        "chat_template_sha256": stable_hash(tokenizer.chat_template),
        "model_placement": placement,
        "hook_layer": layer_index,
        "hook_smoke_records": len(smoke),
        "record_count": len(predictions),
        "macro_family_accuracy": float(np.mean(list(family_accuracy.values()))),
        "family_accuracy": family_accuracy,
        "parse_failure_rate": float(np.mean([not row["parse_ok"] for row in predictions])),
        "median_latency_seconds": float(statistics.median(row["latency_seconds"] for row in predictions)),
        "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in [*smoke, *predictions]),
        "hard_gates": {
            "loaded_on_cuda": placement["loaded_on_cuda"],
            "residual_read": all(row["hook"]["calls"] > 0 for row in smoke),
            "residual_modify": all(row["hook"]["modified"] for row in smoke),
            "hook_smoke_complete": len(smoke) == smoke_count,
            "hook_smoke_finite": all(row["hook"]["finite"] for row in smoke),
            "qualification_complete": len(predictions) == int(charter["qualification"]["records"]),
        },
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "stage0_manifest_sha256": access_validation["source_stage0_manifest_sha256"],
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "readiness_sha256": sha256_file(readiness_path),
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
        "artifacts": {
            "run_intent": {"path": str(intent_path), "sha256": sha256_file(intent_path)},
            "access_view_manifest": {"path": str(access_manifest_path.resolve()), "sha256": sha256_file(access_manifest_path)},
            "hook_smoke": {"path": str(smoke_path), "sha256": sha256_file(smoke_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        },
    }
    manifest["hard_gate_pass"] = all(manifest["hard_gates"].values())
    manifest["run_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    _write_progress(progress_path, candidate_id, "complete", len(predictions), len(prompts))
    return manifest
