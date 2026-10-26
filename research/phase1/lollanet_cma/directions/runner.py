from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from ..access_views.views import validate_access_view
from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..qualification.hooks import transformer_layers
from ..qualification.parser import parse_yes_no
from ..qualification.runner import _model_placement
from .fitting import balanced_record_ids, build_normalized_directions, eligibility_counts, layer_indices
from .hooks import capture_residuals
from .preflight import _verified_content, _verify_checkpoint_files


def _render(tokenizer: Any, instruction: str, question: str, answer_protocol: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"{instruction} {answer_protocol}"},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _encode_prefix(tokenizer: Any, rendered: str, max_input_tokens: int) -> dict[str, Any]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    tokens = int(encoded["input_ids"].shape[1])
    if tokens > max_input_tokens:
        raise ValueError(f"input exceeds frozen token limit: {tokens} > {max_input_tokens}")
    return encoded


def _mean_completion_log_probability(
    model: Any,
    tokenizer: Any,
    encoded_prefix: dict[str, Any],
    completion: str,
    target_layers: list[int] | None = None,
) -> tuple[float, dict[int, Any]]:
    import torch

    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if int(completion_ids.shape[1]) < 1:
        raise ValueError("canonical completion tokenized to an empty sequence")
    input_device = model.get_input_embeddings().weight.device
    prefix_ids = encoded_prefix["input_ids"].to(input_device)
    completion_ids = completion_ids.to(input_device)
    input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    prefix_length = int(prefix_ids.shape[1])
    layers = target_layers or []
    with torch.inference_mode(), capture_residuals(model, layers, prefix_length - 1) as captured:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    positions = logits[:, prefix_length - 1:-1, :]
    log_probabilities = torch.log_softmax(positions.float(), dim=-1)
    token_log_probabilities = log_probabilities.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    score = float(token_log_probabilities.mean().item())
    if layers and set(captured) != set(layers):
        raise RuntimeError("not every frozen layer produced an activation")
    return score, captured


def _score_condition(
    model: Any,
    tokenizer: Any,
    rendered: str,
    gold_answer: str,
    scoring: dict[str, Any],
    target_layers: list[int],
    max_input_tokens: int,
) -> dict[str, Any]:
    encoded = _encode_prefix(tokenizer, rendered, max_input_tokens)
    yes_text = str(scoring["positive_completion"])
    no_text = str(scoring["negative_completion"])
    yes_score, captured = _mean_completion_log_probability(
        model, tokenizer, encoded, yes_text, target_layers,
    )
    no_score, _ = _mean_completion_log_probability(model, tokenizer, encoded, no_text)
    gold_score, other_score = (yes_score, no_score) if gold_answer == "yes" else (no_score, yes_score)
    return {
        "yes_mean_log_probability": yes_score,
        "no_mean_log_probability": no_score,
        "gold_margin": gold_score - other_score,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "activations": captured,
    }


def _generate_positive(
    model: Any, tokenizer: Any, rendered: str, max_input_tokens: int, max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    encoded = _encode_prefix(tokenizer, rendered, max_input_tokens)
    input_tokens = int(encoded["input_ids"].shape[1])
    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, input_tokens:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    return {
        "generated_text": text,
        "output_tokens": int(continuation.shape[0]),
        **parse_yes_no(text),
    }


def _atomic_save_tensors(path: Path, tensors: dict[str, Any], metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, temporary, metadata=metadata)
    temporary.replace(path)


def _load_prefix(path: Path, expected_ids: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids[:len(rows)] or len(observed) != len(set(observed)):
        raise ValueError(f"{label} partial artifact is not an exact ordered prefix")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content):
            raise ValueError(f"{label} record content hash mismatch: {row.get('record_id')}")
        contrast = Path(str(row.get("contrast_path", "")))
        if not contrast.is_file() or sha256_file(contrast) != row.get("contrast_sha256"):
            raise ValueError(f"{label} contrast artifact is missing or changed: {row.get('record_id')}")
    return rows


def _write_progress(path: Path, stage: str, completed: int, total: int) -> None:
    atomic_write_json(path, {
        "type": "lollanet_cma_direction_progress_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "completed": completed,
        "total": total,
    })


def _selected_run(selection: dict[str, Any]) -> dict[str, Any]:
    selected_id = selection["selected_candidate_id"]
    for source in selection["sources"]:
        path = Path(source["manifest_path"])
        if not path.is_file() or sha256_file(path) != source.get("manifest_sha256"):
            raise ValueError("selected qualification source manifest is missing or changed")
        manifest = read_json(path)
        if manifest.get("candidate_id") == selected_id:
            verified = _verified_content(manifest, "run_content_sha256", "selected qualification run")
            if (
                verified.get("model_id") != selection.get("selected_model_id")
                or verified.get("revision") != selection.get("selected_revision")
                or not verified.get("hard_gate_pass")
                or not all(verified.get("hard_gates", {}).values())
            ):
                raise ValueError("selected qualification run is not eligible or has a different identity")
            return verified
    raise ValueError("selected qualification run manifest is absent")


def run_direction_fitting(
    charter_path: Path,
    access_view: Path,
    selection_path: Path,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from safetensors.numpy import load_file as load_numpy_tensors
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    selection = _verified_content(
        read_json(selection_path), "qualification_content_sha256", "qualification selection",
    )
    preflight = _verified_content(read_json(preflight_path), "preflight_content_sha256", "direction preflight")
    if (
        selection.get("type") != "lollanet_cma_base_model_qualification_v1"
        or selection.get("status") != "frozen_before_vector_training"
    ):
        raise ValueError("qualification selection has an unexpected type or status")
    if preflight.get("type") != "lollanet_cma_direction_preflight_v1":
        raise ValueError("direction preflight has an unexpected type")
    if preflight.get("code") != runtime_code_identity(Path(__file__).parent):
        raise ValueError("direction code differs from preflight")
    if preflight.get("dependency_lock") != dependency_lock():
        raise ValueError("direction dependency lock differs from preflight")
    if preflight.get("environment") != runtime_environment():
        raise ValueError("direction runtime differs from preflight")
    if preflight.get("direction_gate_status") != "ready" or preflight.get("errors"):
        raise ValueError("direction preflight is not ready")
    if preflight.get("charter", {}).get("sha256") != sha256_file(charter_path):
        raise ValueError("direction preflight used a different charter")
    access_manifest_path = access_view / "manifest.json"
    if preflight.get("access_view", {}).get("manifest_sha256") != sha256_file(access_manifest_path):
        raise ValueError("direction preflight used a different access view")
    if preflight.get("access_view", {}).get("path") != str(access_view.resolve()):
        raise ValueError("direction access-view path differs from preflight")
    if preflight.get("access_view", {}).get("validator_code") != runtime_code_identity(Path(__file__).parents[1] / "access_views"):
        raise ValueError("direction access-view validator code differs from preflight")
    access_validation = validate_access_view(access_view, "vector_train")
    if preflight.get("access_view", {}).get("vector_train", {}).get("validation") != access_validation:
        raise ValueError("direction access view no longer matches preflight")
    if preflight.get("qualification_selection", {}).get("sha256") != sha256_file(selection_path):
        raise ValueError("direction preflight used a different qualification selection")

    selected_run = _selected_run(selection)
    if selected_run.get("environment") != runtime_environment():
        raise ValueError("direction runtime differs from the selected qualification runtime")
    if selected_run.get("dependency_lock") != dependency_lock():
        raise ValueError("direction dependency lock differs from qualification")
    checkpoint = Path(preflight["qualification_selection"]["tokenizer_prompt_length_audit"]["checkpoint"])
    checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
    if sha256_file(checkpoint_manifest) != preflight["qualification_selection"]["tokenizer_prompt_length_audit"]["checkpoint_manifest_sha256"]:
        raise ValueError("selected checkpoint differs from direction preflight")
    checkpoint_value = _verify_checkpoint_files(checkpoint)
    if checkpoint_value["checkpoint_content_sha256"] != preflight["qualification_selection"]["tokenizer_prompt_length_audit"]["checkpoint_content_sha256"]:
        raise ValueError("selected checkpoint content differs from direction preflight")

    access_manifest = read_json(access_manifest_path)
    artifacts = access_manifest["artifacts"]
    prompts = read_jsonl(Path(artifacts["prompts"]["path"]))
    controller = {row["record_id"]: row for row in read_jsonl(Path(artifacts["controller"]["path"]))}
    evaluator = {row["record_id"]: row for row in read_jsonl(Path(artifacts["evaluator"]["path"]))}
    if len(prompts) != int(charter["data"]["expected_records"]):
        raise ValueError("direction input does not match the frozen vector_train count")

    config = charter["execution"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "fit_records.jsonl"
    progress_path = output_dir / "progress.json"
    intent_path = output_dir / "run_intent.json"
    contrasts_dir = output_dir / "contrasts"
    expected_ids = [str(row["record_id"]) for row in prompts]

    set_seed(int(config["seed"]))
    dtype = torch.bfloat16 if selected_run["compute_dtype"] == "bfloat16" else torch.float16
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
        raise RuntimeError(f"selected model was not loaded entirely on CUDA: {placement}")
    target_layers = layer_indices(len(transformer_layers(model)), list(charter["direction"]["layer_fractions"]))
    intent = {
        "type": "lollanet_cma_direction_run_intent_v1",
        "selected_candidate_id": selection["selected_candidate_id"],
        "model_id": selection["selected_model_id"],
        "revision": selection["selected_revision"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "quantization": selected_run["quantization"],
        "compute_dtype": selected_run["compute_dtype"],
        "maximum_input_tokens": int(config["maximum_input_tokens"]),
        "maximum_new_tokens": int(config["maximum_new_tokens"]),
        "sampling": str(config["sampling"]),
        "seed": int(config["seed"]),
        "charter_sha256": sha256_file(charter_path),
        "stage0_manifest_sha256": access_validation["source_stage0_manifest_sha256"],
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "preflight_sha256": sha256_file(preflight_path),
        "chat_template_sha256": stable_hash(tokenizer.chat_template),
        "target_layers": target_layers,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
    }
    intent["intent_content_sha256"] = stable_hash(intent)
    if intent_path.is_file():
        if read_json(intent_path) != intent:
            raise ValueError("existing direction run intent differs from the frozen contract")
    else:
        atomic_write_json(intent_path, intent)

    rows = _load_prefix(rows_path, expected_ids, "direction fitting")
    answer_protocol = str(charter["prompt_contract"]["answer_protocol"])
    instructions = charter["prompt_contract"]["instructions"]
    scoring = charter["prompt_contract"]["margin_scoring"]
    mapping = charter["contrast_mapping"]
    max_input_tokens = int(config["maximum_input_tokens"])
    max_new_tokens = int(config["maximum_new_tokens"])
    for prompt_row in prompts[len(rows):]:
        record_id = str(prompt_row["record_id"])
        family = str(controller[record_id]["model_family"])
        negative_family = str(mapping[family])
        gold_answer = str(evaluator[record_id]["answer"]).lower()
        positive_rendered = _render(tokenizer, instructions[family], prompt_row["prompt"], answer_protocol)
        negative_rendered = _render(tokenizer, instructions[negative_family], prompt_row["prompt"], answer_protocol)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        positive = _score_condition(
            model, tokenizer, positive_rendered, gold_answer, scoring, target_layers, max_input_tokens,
        )
        negative = _score_condition(
            model, tokenizer, negative_rendered, gold_answer, scoring, target_layers, max_input_tokens,
        )
        generated = _generate_positive(model, tokenizer, positive_rendered, max_input_tokens, max_new_tokens)
        torch.cuda.synchronize()
        contrasts = {
            f"layer_{layer_id}": (positive["activations"][layer_id] - negative["activations"][layer_id]).float().contiguous()
            for layer_id in target_layers
        }
        hidden_finite = all(bool(torch.isfinite(vector).all().item()) for vector in contrasts.values())
        contrast_path = contrasts_dir / f"{stable_hash(record_id)[:24]}.safetensors"
        _atomic_save_tensors(contrast_path, contrasts, {"record_id": record_id})
        row = {
            "record_id": record_id,
            "model_family": family,
            "negative_family": negative_family,
            "gold_answer": gold_answer,
            "positive_parse_ok": generated["parse_ok"],
            "positive_parsed_answer": generated["parsed_answer"],
            "positive_parse_rule": generated["parse_rule"],
            "positive_correct": bool(generated["parse_ok"] and generated["parsed_answer"] == gold_answer),
            "positive_generated_text": generated["generated_text"],
            "positive_output_tokens": generated["output_tokens"],
            "positive_yes_mean_log_probability": positive["yes_mean_log_probability"],
            "positive_no_mean_log_probability": positive["no_mean_log_probability"],
            "positive_gold_margin": positive["gold_margin"],
            "negative_yes_mean_log_probability": negative["yes_mean_log_probability"],
            "negative_no_mean_log_probability": negative["no_mean_log_probability"],
            "negative_gold_margin": negative["gold_margin"],
            "positive_input_tokens": positive["input_tokens"],
            "negative_input_tokens": negative["input_tokens"],
            "hidden_finite": hidden_finite,
            "contrast_norms": {key: float(torch.linalg.vector_norm(value).item()) for key, value in contrasts.items()},
            "contrast_path": str(contrast_path.resolve()),
            "contrast_sha256": sha256_file(contrast_path),
            "latency_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        row["record_content_sha256"] = stable_hash(row)
        rows.append(row)
        atomic_write_jsonl(rows_path, rows)
        _write_progress(progress_path, "fit_records", len(rows), len(prompts))

    families = list(charter["families"])
    labels = list(charter["filtering"]["labels"])
    selected_ids, balance = balanced_record_ids(rows, families, labels, int(charter["filtering"]["seed"]))
    minimum = int(charter["filtering"]["minimum_balanced_records_per_family"])
    sufficient = all(len(record_ids) >= minimum for record_ids in selected_ids.values())
    artifacts_out: dict[str, Any] = {
        "run_intent": {"path": str(intent_path), "sha256": sha256_file(intent_path)},
        "fit_records": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "selection": {"path": str(selection_path.resolve()), "sha256": sha256_file(selection_path)},
        "preflight": {"path": str(preflight_path.resolve()), "sha256": sha256_file(preflight_path)},
        "access_view_manifest": {
            "path": str(access_manifest_path.resolve()),
            "sha256": sha256_file(access_manifest_path),
        },
        "stage0_manifest": {
            "path": str(Path(access_manifest["source_stage0"]["manifest_path"]).resolve()),
            "sha256": access_validation["source_stage0_manifest_sha256"],
        },
        "checkpoint_manifest": {
            "path": str(checkpoint_manifest.resolve()),
            "sha256": sha256_file(checkpoint_manifest),
        },
    }
    direction_metadata: dict[str, Any] | None = None
    if sufficient:
        row_by_id = {row["record_id"]: row for row in rows}

        def load_contrast(record_id: str, layer_id: int) -> np.ndarray:
            source = row_by_id[record_id]
            if sha256_file(Path(source["contrast_path"])) != source["contrast_sha256"]:
                raise ValueError(f"contrast changed before aggregation: {record_id}")
            return load_numpy_tensors(source["contrast_path"])[f"layer_{layer_id}"]

        vectors, direction_metadata = build_normalized_directions(selected_ids, target_layers, load_contrast)
        tensor_vectors = {key: torch.from_numpy(value) for key, value in vectors.items()}
        directions_path = output_dir / "directions.safetensors"
        temporary = directions_path.with_suffix(".safetensors.tmp")
        save_file(tensor_vectors, temporary, metadata={"dtype": "float32"})
        temporary.replace(directions_path)
        index = {
            "type": "lollanet_cma_direction_index_v1",
            "model_id": selection["selected_model_id"],
            "revision": selection["selected_revision"],
            "target_layers": target_layers,
            "selected_record_ids": selected_ids,
            "balance": balance,
            "direction_metadata": direction_metadata,
            "directions_sha256": sha256_file(directions_path),
        }
        index["direction_index_content_sha256"] = stable_hash(index)
        index_path = output_dir / "direction_index.json"
        atomic_write_json(index_path, index)
        artifacts_out["directions"] = {"path": str(directions_path), "sha256": sha256_file(directions_path)}
        artifacts_out["direction_index"] = {"path": str(index_path), "sha256": sha256_file(index_path)}

    manifest = {
        "type": "lollanet_cma_direction_fit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "fitted" if sufficient else "insufficient_balanced_records",
        "model_id": selection["selected_model_id"],
        "revision": selection["selected_revision"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "quantization": selected_run["quantization"],
        "compute_dtype": selected_run["compute_dtype"],
        "maximum_input_tokens": int(config["maximum_input_tokens"]),
        "maximum_new_tokens": int(config["maximum_new_tokens"]),
        "sampling": str(config["sampling"]),
        "seed": int(config["seed"]),
        "chat_template_sha256": stable_hash(tokenizer.chat_template),
        "target_layers": target_layers,
        "record_count": len(rows),
        "eligibility_counts": eligibility_counts(rows, families),
        "balance": balance,
        "selected_record_counts": {family: len(ids) for family, ids in selected_ids.items()},
        "direction_metadata": direction_metadata,
        "hard_gates": {
            "record_coverage_complete": len(rows) == int(charter["data"]["expected_records"]),
            "all_hidden_states_finite": all(row["hidden_finite"] for row in rows),
            "all_families_sufficient": sufficient,
            "directions_created": sufficient and "directions" in artifacts_out,
        },
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "stage0_manifest_sha256": access_validation["source_stage0_manifest_sha256"],
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "preflight_sha256": sha256_file(preflight_path),
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
        "model_placement": placement,
        "artifacts": artifacts_out,
    }
    manifest["hard_gate_pass"] = all(manifest["hard_gates"].values())
    manifest["direction_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    _write_progress(progress_path, "complete", len(rows), len(prompts))
    return manifest
