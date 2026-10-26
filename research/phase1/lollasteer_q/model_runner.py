from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterator

import numpy as np

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, runtime_code_identity, sha256_file, stable_hash
from .prompts import OPTION_LABELS, prompt_only_instruction
from .runtime_audit import audit_direction_manifest
from .steering import Intervention

ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(?:<option\s+letter>)?\s*([A-J])\s*>?\b")
CONFIDENCE_RE = re.compile(r"(?im)^\s*Confidence\s*:\s*(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%?")


def parse_generation(text: str) -> dict[str, Any]:
    answer = ANSWER_RE.search(text)
    confidence = CONFIDENCE_RE.search(text)
    confidence_value = float(confidence.group(1)) / 100.0 if confidence else None
    return {
        "parsed_answer": answer.group(1) if answer else None,
        "self_reported_confidence": confidence_value,
        "parse_ok": bool(answer and confidence and confidence_value is not None and 0 <= confidence_value <= 1),
    }


def relative_layer_indices(layer_count: int, fractions: list[float]) -> list[int]:
    if layer_count < 1:
        raise ValueError("model has no transformer layers")
    return sorted({min(layer_count - 1, max(0, round((layer_count - 1) * float(value)))) for value in fractions})


def _layers(model: Any) -> Any:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for value in candidates:
        if value is not None:
            return value
    raise TypeError("unsupported causal-LM layer layout")


def _runtime_environment(torch: Any, transformers: Any, model: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    raw_device_map = getattr(model, "hf_device_map", None)
    device_map = None if raw_device_map is None else {
        str(name): str(device) for name, device in sorted(raw_device_map.items(), key=lambda item: str(item[0]))
    }
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "transformers": str(transformers.__version__),
        "numpy": str(np.__version__), "cuda_runtime": str(torch.version.cuda),
        "cuda_available": cuda_available, "cuda_device_count": cuda_device_count,
        "cuda_device_names": [str(torch.cuda.get_device_name(index)) for index in range(cuda_device_count)],
        "model_device": str(model.device), "model_device_map": device_map,
        "python_executable": str(Path(sys.executable).resolve()),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("softmax requires a finite non-empty vector")
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def score_answer_prefixes(model: Any, tokenizer: Any, rendered_prompt: str, option_count: int, intervention: Intervention | None) -> np.ndarray:
    """Score `Answer: X` token by token while steering only the current final position."""
    import torch
    if not 2 <= option_count <= len(OPTION_LABELS):
        raise ValueError("option_count must be between 2 and the supported label count")
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"].to(model.device)
    scores: list[float] = []
    hook_layers = [intervention.layer_index] if intervention is not None else []
    for label in OPTION_LABELS[:option_count]:
        continuation = tokenizer("Answer: " + label, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        if continuation.ndim != 2 or continuation.shape[1] < 1:
            raise ValueError("answer continuation tokenization is empty or malformed")
        score = 0.0
        for token_position in range(continuation.shape[1]):
            prefix = continuation[:, :token_position]
            input_ids = torch.cat([prompt_ids, prefix], dim=1)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode(), residual_hooks(model, hook_layers, intervention):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            target = continuation[0, token_position]
            token_log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
            score += float(token_log_probs[target].cpu())
        scores.append(score)
    result = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("answer-prefix score is non-finite")
    return result


@contextmanager
def residual_hooks(model: Any, layer_indices: list[int], intervention: Intervention | None) -> Iterator[dict[int, np.ndarray]]:
    captured: dict[int, np.ndarray] = {}
    handles = []
    layers = _layers(model)

    def hook_for(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[0, -1].detach().float().cpu().numpy()
            if intervention is None or intervention.layer_index != layer_index:
                return output
            delta = intervention.delta(captured[layer_index])
            if np.asarray(delta).shape != captured[layer_index].shape or not np.all(np.isfinite(delta)):
                raise ValueError("intervention delta must be finite and match the hidden size")
            delta_tensor = torch.as_tensor(delta, device=hidden.device, dtype=hidden.dtype)
            modified = hidden.clone()
            modified[:, -1, :] = modified[:, -1, :] + delta_tensor.view(1, -1)
            return (modified, *output[1:]) if isinstance(output, tuple) else modified
        return hook

    try:
        for layer_index in layer_indices:
            handles.append(layers[layer_index].register_forward_hook(hook_for(layer_index)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def oracle_factor(row: dict[str, Any]) -> str | None:
    control = str(row.get("control_type"))
    valid_control_factor = {"valid_reward": "reward", "valid_social": "social", "calibration": "certainty"}
    if control in valid_control_factor:
        return valid_control_factor[control]
    if control != "factorial":
        return None
    active = [
        factor for factor, field in (("reward", "reward_lure"), ("social", "social_lure"), ("certainty", "certainty_pressure"))
        if int(row.get(field, 0)) == 1
    ]
    return active[0] if len(active) == 1 else None


def conditioned_prompt(prompt: str, method: str, factor: str | None, apply_oracle: bool) -> str:
    if method != "prompt_only" or not apply_oracle:
        return prompt
    if factor is None:
        raise ValueError("prompt-only conditioning requires a factor")
    return f"{prompt}\n\nAdditional evidence rule:\n{prompt_only_instruction(factor)}"


def run_model(
    prompts_path: Path,
    output_dir: Path,
    checkpoint: str,
    fallback_checkpoint: str | None,
    fallback_policy: str,
    layer_fractions: list[float],
    max_new_tokens: int,
    dtype_name: str,
    direction_path: Path | None = None,
    method: str = "baseline",
    selected_factor: str | None = None,
    selected_layer: int | None = None,
    strength: float = 1.0,
    rank: int = 1,
    limit: int | None = None,
    checkpoint_revision: str | None = None,
    local_checkpoint: str | None = None,
    split: str = "all",
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("run inside the locked lollasteer-q Python 3.11 environment") from exc
    all_rows = read_jsonl(prompts_path)
    if split not in {"all", "discovery", "pilot_holdout"}:
        raise ValueError("split must be all, discovery, or pilot_holdout")
    rows = all_rows if split == "all" else [row for row in all_rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"prompt file contains no records for split {split}")
    if limit is not None:
        rows = rows[:limit]
    if method not in {"baseline", "prompt_only", "caa", "low_rank"}:
        raise ValueError(f"unknown method: {method}")
    if method == "baseline" and any(value is not None for value in (direction_path, selected_factor, selected_layer)):
        raise ValueError("baseline does not accept direction, factor, or layer")
    if method == "prompt_only" and selected_factor is None:
        raise ValueError("prompt-only steering requires a factor")
    if method == "prompt_only" and any(value is not None for value in (direction_path, selected_layer)):
        raise ValueError("prompt-only steering does not accept a direction or layer")
    if method in {"caa", "low_rank"} and (direction_path is None or selected_factor is None or selected_layer is None):
        raise ValueError("linear steering requires direction path, factor, and layer")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    actual_checkpoint = checkpoint
    checkpoint_source = checkpoint
    local_files_only = False
    if local_checkpoint is not None:
        local_path = Path(local_checkpoint)
        if not local_path.is_dir():
            raise FileNotFoundError(f"frozen local checkpoint is missing: {local_path}")
        metadata_path = local_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
        if checkpoint_revision is None or not metadata_path.is_file():
            raise ValueError("local checkpoint requires revision and Hugging Face revision metadata")
        local_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if local_revision != checkpoint_revision:
            raise ValueError(f"local checkpoint revision mismatch: {local_revision} != {checkpoint_revision}")
        checkpoint_source = str(local_path)
        local_files_only = True
    fallback_used = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_source, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_source, dtype=dtype, device_map="auto", local_files_only=local_files_only,
        )
    except (OSError, ValueError) as primary_error:
        if fallback_policy != "license_or_download_failure_only" or not fallback_checkpoint:
            raise
        actual_checkpoint = fallback_checkpoint
        fallback_used = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_checkpoint)
            model = AutoModelForCausalLM.from_pretrained(fallback_checkpoint, dtype=dtype, device_map="auto")
        except Exception:
            raise primary_error
    model.eval()
    lock_path = Path(__file__).with_name("uv.lock")
    runtime_code = runtime_code_identity(Path(__file__).parent)
    runtime_environment = _runtime_environment(torch, transformers, model)
    runtime_environment_sha256 = stable_hash(runtime_environment)
    dependency_lock_sha256 = sha256_file(lock_path)
    layers = _layers(model)
    layer_indices = relative_layer_indices(len(layers), layer_fractions)
    if selected_layer is not None and selected_layer not in layer_indices:
        raise ValueError("selected steering layer is outside preregistered layers")
    intervention_template = None
    direction_source = None
    if method in {"caa", "low_rank"}:
        direction_metadata_path = direction_path.with_suffix(".json")
        if not direction_metadata_path.is_file():
            raise ValueError("linear steering requires direction bundle metadata")
        import json
        direction_metadata = json.loads(direction_metadata_path.read_text(encoding="utf-8"))
        if audit_direction_manifest(direction_metadata_path).get("status") != "current_eligible":
            raise ValueError("linear steering requires a current audited direction bundle")
        if direction_metadata.get("npz_sha256") != sha256_file(direction_path):
            raise ValueError("direction bundle hash does not match metadata")
        if direction_metadata.get("checkpoint_actual") != actual_checkpoint:
            raise ValueError("direction bundle checkpoint does not match loaded model")
        if direction_metadata.get("checkpoint_revision") != checkpoint_revision:
            raise ValueError("direction bundle revision does not match loaded model")
        if direction_metadata.get("chat_template_sha256") != stable_hash(tokenizer.chat_template):
            raise ValueError("direction bundle chat template does not match the current run")
        if direction_metadata.get("dtype") != dtype_name:
            raise ValueError("direction bundle dtype does not match the current run")
        if direction_metadata.get("dependency_lock_sha256") != dependency_lock_sha256:
            raise ValueError("direction bundle dependency lock does not match the current run")
        if direction_metadata.get("runtime_environment_sha256") != runtime_environment_sha256:
            raise ValueError("direction bundle runtime environment does not match the current run")
        if direction_metadata.get("prompt_bundle_sha256") != sha256_file(prompts_path):
            raise ValueError("direction bundle prompt source does not match the current run")
        if [int(value) for value in direction_metadata.get("layer_indices", [])] != layer_indices:
            raise ValueError("direction bundle layers do not match the current run")
        factor_metadata = direction_metadata.get("factors", {}).get(selected_factor, {}).get(str(selected_layer), {})
        if int(factor_metadata.get("available_rank", 0)) < int(rank):
            raise ValueError("requested rank exceeds direction bundle numerical rank")
        configured_hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if configured_hidden_size is not None and int(factor_metadata.get("hidden_size", -1)) != int(configured_hidden_size):
            raise ValueError("direction bundle hidden size does not match the loaded model")
        intervention_template = Intervention(direction_path, method, selected_factor, selected_layer, strength, rank)
        direction_source = {
            "path": str(direction_path), "sha256": sha256_file(direction_path),
            "metadata_path": str(direction_metadata_path), "metadata_sha256": sha256_file(direction_metadata_path),
            "source_run": direction_metadata.get("source_run"),
        }
    run_key = "baseline" if method == "baseline" else (
        f"prompt_only__{selected_factor}" if method == "prompt_only"
        else f"{method}__{selected_factor}__layer_{selected_layer}__strength_{strength:g}__rank_{rank}"
    )
    prompt_policy_hash = stable_hash(prompt_only_instruction(selected_factor)) if method == "prompt_only" else None
    outputs: list[dict[str, Any]] = []
    activation_arrays: dict[str, np.ndarray] = {}
    is_linear = method in {"caa", "low_rank"}
    for row in rows:
        apply_oracle = selected_factor is not None and oracle_factor(row) == selected_factor
        intervention = intervention_template if apply_oracle else None
        prompt_text = conditioned_prompt(str(row["prompt"]), method, selected_factor, apply_oracle)
        messages = [{"role": "user", "content": prompt_text}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            # Older compatible chat templates do not expose Qwen3's switch.
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode(), residual_hooks(model, layer_indices, intervention) as captured:
            model(**encoded, use_cache=False)
        for layer_index, activation in captured.items():
            activation_arrays[f"{row['record_id']}__layer_{layer_index}"] = activation.astype(np.float32)
        option_logits = score_answer_prefixes(model, tokenizer, rendered, len(row["options"]), intervention)
        probabilities = _softmax(option_logits.astype(np.float64))
        with torch.inference_mode(), residual_hooks(model, layer_indices, intervention):
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append({
            "record_id": row["record_id"], "base_item_id": row["base_item_id"], "split": row["split"],
            "category": row["category"], "template_family": row["template_family"],
            "certainty_keyword_free": bool(row.get("certainty_keyword_free", False)),
            "reward_lure": row["reward_lure"], "social_lure": row["social_lure"],
            "certainty_pressure": row["certainty_pressure"], "control_type": row["control_type"],
            "method": method, "run_key": run_key, "steering_factor": selected_factor,
            "steering_layer": selected_layer if is_linear else None,
            "steering_strength": strength if is_linear else None,
            "steering_rank": rank if is_linear else None,
            "oracle_intervention_applied": apply_oracle,
            "model_input_prompt_sha256": stable_hash(prompt_text),
            "option_logits": option_logits.tolist(), "option_probabilities": probabilities.tolist(),
            "generated_text": completion, **parse_generation(completion),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "outputs.jsonl"
    activations_path = output_dir / "activations.npz"
    atomic_write_jsonl(outputs_path, outputs)
    np.savez_compressed(activations_path, **activation_arrays)
    execution_identity = {
        "prompt_bundle_sha256": sha256_file(prompts_path),
        "record_ids_sha256": stable_hash(sorted(str(row["record_id"]) for row in rows)),
        "requested_split": split, "checkpoint_actual": actual_checkpoint,
        "checkpoint_revision": checkpoint_revision, "dtype": dtype_name,
        "chat_template_sha256": stable_hash(tokenizer.chat_template),
        "method": method, "factor": selected_factor, "layer": selected_layer if is_linear else None,
        "strength": strength if is_linear else None, "rank": rank if is_linear else None,
        "direction_bundle_sha256": direction_source["sha256"] if direction_source else None,
        "prompt_only_instruction_sha256": prompt_policy_hash,
        "max_new_tokens": max_new_tokens, "do_sample": False, "batch_size": 1,
        "oracle_gate_policy": "single_active_factor_plus_matching_valid_control",
        "dependency_lock_sha256": dependency_lock_sha256,
        "runtime_code_sha256": runtime_code["sha256"],
        "runtime_environment_sha256": runtime_environment_sha256,
        "activation_capture_policy": "last_prompt_token_pre_intervention",
        "intervention_position_policy": "last_sequence_token_each_forward",
        "option_scoring_policy": "teacher_forced_answer_prefix_token_by_token",
    }
    execution_identity_sha256 = stable_hash(execution_identity)
    run_id = execution_identity_sha256[:24]
    manifest = {
        "type": "lollasteer_q_model_run", "run_id": run_id, "checkpoint_requested": checkpoint,
        "checkpoint_actual": actual_checkpoint, "checkpoint_revision": checkpoint_revision,
        "checkpoint_load_path": checkpoint_source, "local_files_only": local_files_only,
        "fallback_used": fallback_used, "dtype": dtype_name,
        "chat_template_sha256": stable_hash(tokenizer.chat_template), "layer_indices": layer_indices,
        "method": method, "run_key": run_key, "factor": selected_factor,
        "layer": selected_layer if is_linear else None,
        "strength": strength if is_linear else None,
        "rank": rank if is_linear else None,
        "requested_split": split,
        "execution_identity_sha256": execution_identity_sha256,
        "dependency_lock": {"path": str(lock_path.resolve()), "sha256": dependency_lock_sha256},
        "runtime_code": runtime_code,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256,
        "record_ids_sha256": stable_hash(sorted(str(row["record_id"]) for row in rows)),
        "direction_bundle": direction_source,
        "prompt_only_instruction_sha256": prompt_policy_hash,
        "oracle_gate_policy": "single_active_factor_plus_matching_valid_control",
        "do_sample": False, "batch_size": 1, "max_new_tokens": max_new_tokens,
        "activation_capture_policy": "last_prompt_token_pre_intervention",
        "intervention_position_policy": "last_sequence_token_each_forward",
        "option_scoring_policy": "teacher_forced_answer_prefix_token_by_token",
        "inputs": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
        "artifacts": {
            "outputs": {"path": str(outputs_path), "sha256": sha256_file(outputs_path)},
            "activations": {"path": str(activations_path), "sha256": sha256_file(activations_path)},
        },
        "record_count": len(outputs),
        "prompt_record_count": len(all_rows),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return manifest
