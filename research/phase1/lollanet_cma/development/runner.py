from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np
import yaml

from ..access_views.views import validate_access_view
from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..directions.fitting import _array_sha256
from ..directions.preflight import _verify_checkpoint_files
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..qualification.parser import parse_yes_no
from ..qualification.runner import SYSTEM_INSTRUCTION, _model_placement
from .hooks import steer_residual
from .selection import choose_grid_setting, specificity_result, summarize_condition


def _verified_content(value: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    content = dict(value)
    claimed = content.pop(field, None)
    if claimed != stable_hash(content):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _load_condition_prefix(path: Path, expected_ids: list[str], condition_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids[:len(rows)] or len(observed) != len(set(observed)):
        raise ValueError(f"development condition is not an exact ordered prefix: {condition_id}")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content) or row.get("condition_id") != condition_id:
            raise ValueError(f"development condition row changed: {condition_id} {row.get('record_id')}")
    return rows


def _condition_id(kind: str, layer: int | None = None, alpha: float | None = None) -> str:
    if layer is None or alpha is None:
        return kind
    alpha_text = format(float(alpha), "g").replace(".", "p")
    return f"{kind}__layer_{layer}__alpha_{alpha_text}"


def _vector_reference(
    kind: str,
    family: str,
    layer: int,
    wrong_mapping: dict[str, str],
) -> tuple[str, str]:
    if kind == "oracle_correct":
        tensor_key = f"{family}__layer_{layer}"
        return tensor_key, tensor_key
    if kind == "oracle_wrong":
        tensor_key = f"{wrong_mapping[family]}__layer_{layer}"
        return tensor_key, tensor_key
    if kind == "random_norm_matched":
        tensor_key = f"{family}__layer_{layer}"
        return f"random::{tensor_key}", tensor_key
    raise ValueError(f"unsupported development steering condition: {kind}")


def _render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    layer: int | None,
    vector: Any | None,
    alpha: float | None,
) -> dict[str, Any]:
    import torch

    rendered = _render(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        raise ValueError(f"development input exceeds the frozen token limit: {input_tokens}")
    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    context = (
        steer_residual(model, int(layer), vector, float(alpha))
        if layer is not None and vector is not None and alpha is not None
        else nullcontext(None)
    )
    torch.cuda.reset_peak_memory_stats()
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
    continuation = generated[0, input_tokens:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    if hook_state is not None and (
        int(hook_state["calls"]) < 1 or not hook_state["finite"]
    ):
        raise RuntimeError("development steering hook failed")
    return {
        "generated_text": text,
        "input_tokens": input_tokens,
        "output_tokens": int(continuation.shape[0]),
        "latency_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "hook_calls": 0 if hook_state is None else int(hook_state["calls"]),
        "hook_finite": True if hook_state is None else bool(hook_state["finite"]),
        **parse_yes_no(text),
    }


def _run_condition(
    condition_id: str,
    output_path: Path,
    prompts: list[dict[str, Any]],
    controller: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
    max_input_tokens: int,
    max_new_tokens: int,
    layer: int | None,
    alpha: float | None,
    vector_for_family: Callable[[str], tuple[str | None, Any | None]],
    families: list[str],
    progress_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = [str(row["record_id"]) for row in prompts]
    rows = _load_condition_prefix(output_path, expected_ids, condition_id)
    for prompt_row in prompts[len(rows):]:
        record_id = str(prompt_row["record_id"])
        family = str(controller[record_id]["model_family"])
        vector_key, vector = vector_for_family(family)
        generated = _generate_one(
            model, tokenizer, prompt_row["prompt"], max_input_tokens, max_new_tokens,
            layer, vector, alpha,
        )
        gold = str(evaluator[record_id]["answer"]).lower()
        row = {
            "condition_id": condition_id,
            "record_id": record_id,
            "model_family": family,
            "query_type": str(controller[record_id]["query_type"]),
            "gold_answer": gold,
            "layer": layer,
            "alpha": alpha,
            "vector_key": vector_key,
            "vector_tensor_sha256": None if vector is None else _array_sha256(
                vector.detach().cpu().float().numpy(),
            ),
            "correct": bool(generated["parse_ok"] and generated["parsed_answer"] == gold),
            **generated,
        }
        row["record_content_sha256"] = stable_hash(row)
        rows.append(row)
        atomic_write_jsonl(output_path, rows)
        atomic_write_json(progress_path, {
            "type": "lollanet_cma_development_progress_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "condition",
            "condition_id": condition_id,
            "completed": len(rows),
            "total": len(prompts),
        })
    summary = {
        "condition_id": condition_id,
        "layer": layer,
        "alpha": alpha,
        **summarize_condition(rows, families),
        "median_latency_seconds": float(statistics.median(row["latency_seconds"] for row in rows)),
        "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in rows),
        "artifact": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
    }
    return rows, summary


def _random_vectors(
    families: list[str], layers: list[int], hidden_size: int, seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for layer in layers:
        for family in families:
            value = rng.standard_normal(hidden_size).astype(np.float64)
            norm = float(np.linalg.norm(value))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("random steering vector is non-finite or zero")
            result[f"{family}__layer_{layer}"] = (value / norm).astype(np.float32)
    return result


def _save_random_vectors(path: Path, values: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    tensors = {key: torch.from_numpy(value) for key, value in values.items()}
    if path.is_file():
        observed = load_file(path)
        if set(observed) != set(tensors) or any(
            not torch.equal(observed[key].cpu(), tensors[key]) for key in tensors
        ):
            raise ValueError("existing random direction artifact differs from the frozen seed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, temporary, metadata={"distribution": "standard_normal_unit_norm"})
        temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "tensor_sha256": {key: _array_sha256(value) for key, value in values.items()},
    }


def run_development_selection(
    charter_path: Path,
    access_view: Path,
    direction_manifest_path: Path,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    preflight = _verified_content(
        read_json(preflight_path), "development_preflight_content_sha256", "development preflight",
    )
    if preflight.get("type") != "lollanet_cma_development_preflight_v1":
        raise ValueError("development preflight has an unexpected type")
    if preflight.get("development_gate_status") != "ready" or preflight.get("errors"):
        raise ValueError("development preflight is not ready")
    if preflight.get("charter", {}).get("sha256") != sha256_file(charter_path):
        raise ValueError("development preflight used a different charter")
    if preflight.get("code") != runtime_code_identity(Path(__file__).parent):
        raise ValueError("development code differs from preflight")
    if preflight.get("dependency_lock") != dependency_lock() or preflight.get("environment") != runtime_environment():
        raise ValueError("development dependency lock or runtime differs from preflight")
    access_manifest_path = access_view / "manifest.json"
    if (
        preflight.get("access_view", {}).get("path") != str(access_view.resolve())
        or preflight.get("access_view", {}).get("manifest_sha256") != sha256_file(access_manifest_path)
        or preflight.get("access_view", {}).get("validator_code")
        != runtime_code_identity(Path(__file__).parents[1] / "access_views")
    ):
        raise ValueError("development access view differs from preflight")
    access_validation = validate_access_view(access_view, "development")
    if preflight.get("access_view", {}).get("validation") != access_validation:
        raise ValueError("development access view no longer matches preflight")
    if preflight.get("direction_run", {}).get("manifest_sha256") != sha256_file(direction_manifest_path):
        raise ValueError("development direction manifest differs from preflight")

    direction_manifest = _verified_content(
        read_json(direction_manifest_path), "direction_content_sha256", "direction run",
    )
    directions_path = Path(preflight["direction_run"]["directions_path"])
    if sha256_file(directions_path) != preflight["direction_run"]["directions_sha256"]:
        raise ValueError("development direction tensors differ from preflight")
    directions = load_file(directions_path, device="cpu")
    checkpoint = Path(direction_manifest["checkpoint"])
    checkpoint_value = _verify_checkpoint_files(checkpoint)
    if checkpoint_value["checkpoint_content_sha256"] != direction_manifest["checkpoint_manifest_sha256"]:
        raise ValueError("development checkpoint differs from the direction run")

    access_manifest = read_json(access_manifest_path)
    artifacts = access_manifest["artifacts"]
    prompts = read_jsonl(Path(artifacts["prompts"]["path"]))
    controller = {row["record_id"]: row for row in read_jsonl(Path(artifacts["controller"]["path"]))}
    evaluator = {row["record_id"]: row for row in read_jsonl(Path(artifacts["evaluator"]["path"]))}
    if len(prompts) != int(charter["data"]["expected_records"]):
        raise ValueError("development access view has an unexpected record count")

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir = output_dir / "conditions"
    progress_path = output_dir / "progress.json"
    selection_path = output_dir / "development_selection.json"
    intent_path = output_dir / "run_intent.json"
    random_path = output_dir / "random_vectors.safetensors"
    families = list(charter["families"])
    layers = [int(value) for value in direction_manifest["target_layers"]]
    alphas = [float(value) for value in charter["grid"]["alpha_candidates"]]
    first_vector = directions[f"{families[0]}__layer_{layers[0]}"]
    hidden_size = int(first_vector.numel())
    random_values = _random_vectors(
        families, layers, hidden_size, int(charter["controls"]["random_seed"]),
    )
    random_artifact = _save_random_vectors(random_path, random_values)
    random_tensors = {key: torch.from_numpy(value) for key, value in random_values.items()}

    intent = {
        "type": "lollanet_cma_development_run_intent_v1",
        "model_id": direction_manifest["model_id"],
        "revision": direction_manifest["revision"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_content_sha256": checkpoint_value["checkpoint_content_sha256"],
        "charter_sha256": sha256_file(charter_path),
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "direction_manifest_sha256": sha256_file(direction_manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "direction_tensors_sha256": sha256_file(directions_path),
        "random_vectors": random_artifact,
        "layers": layers,
        "alphas": alphas,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
    }
    intent["intent_content_sha256"] = stable_hash(intent)
    if intent_path.is_file():
        if read_json(intent_path) != intent:
            raise ValueError("existing development intent differs from the frozen contract")
    else:
        atomic_write_json(intent_path, intent)

    set_seed(int(charter["execution"]["seed"]))
    dtype = torch.bfloat16 if direction_manifest["compute_dtype"] == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if stable_hash(tokenizer.chat_template) != direction_manifest["chat_template_sha256"]:
        raise ValueError("development chat template differs from direction fitting")
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
        raise RuntimeError(f"development model was not loaded entirely on CUDA: {placement}")

    max_input_tokens = int(charter["execution"]["maximum_input_tokens"])
    max_new_tokens = int(charter["execution"]["maximum_new_tokens"])
    condition_artifacts: dict[str, Any] = {}
    _, base_summary = _run_condition(
        "base", conditions_dir / "base.jsonl", prompts, controller, evaluator, model, tokenizer,
        max_input_tokens, max_new_tokens, None, None, lambda _family: (None, None), families, progress_path,
    )
    condition_artifacts["base"] = base_summary["artifact"]

    grid_summaries: list[dict[str, Any]] = []
    for layer in layers:
        for alpha in alphas:
            condition_id = _condition_id("oracle_correct", layer, alpha)

            def correct_vector(family: str, layer: int = layer) -> tuple[str, Any]:
                display_key, tensor_key = _vector_reference(
                    "oracle_correct", family, layer, charter["controls"]["wrong_mapping"],
                )
                return display_key, directions[tensor_key]

            _, summary = _run_condition(
                condition_id, conditions_dir / f"{condition_id}.jsonl", prompts, controller, evaluator,
                model, tokenizer, max_input_tokens, max_new_tokens, layer, alpha,
                correct_vector,
                families, progress_path,
            )
            grid_summaries.append(summary)
            condition_artifacts[condition_id] = summary["artifact"]
    chosen = choose_grid_setting(grid_summaries)
    selection = {
        "type": "lollanet_cma_development_grid_selection_v1",
        "rule": "max_macro_family_accuracy_then_lower_alpha_then_shallower_layer",
        "base_summary": base_summary,
        "grid_summaries": grid_summaries,
        "selected_condition_id": chosen["condition_id"],
        "selected_layer": int(chosen["layer"]),
        "selected_alpha": float(chosen["alpha"]),
        "no_fallback_after_specificity_failure": True,
    }
    selection["development_selection_content_sha256"] = stable_hash(selection)
    if selection_path.is_file():
        if read_json(selection_path) != selection:
            raise ValueError("existing development selection differs from completed grid artifacts")
    else:
        atomic_write_json(selection_path, selection)

    layer = int(chosen["layer"])
    alpha = float(chosen["alpha"])
    wrong_mapping = charter["controls"]["wrong_mapping"]
    wrong_id = _condition_id("oracle_wrong", layer, alpha)

    def wrong_vector(family: str) -> tuple[str, Any]:
        display_key, tensor_key = _vector_reference("oracle_wrong", family, layer, wrong_mapping)
        return display_key, directions[tensor_key]

    _, wrong_summary = _run_condition(
        wrong_id, conditions_dir / f"{wrong_id}.jsonl", prompts, controller, evaluator,
        model, tokenizer, max_input_tokens, max_new_tokens, layer, alpha,
        wrong_vector,
        families, progress_path,
    )
    condition_artifacts[wrong_id] = wrong_summary["artifact"]
    random_id = _condition_id("random_norm_matched", layer, alpha)

    def random_vector(family: str) -> tuple[str, Any]:
        display_key, tensor_key = _vector_reference(
            "random_norm_matched", family, layer, wrong_mapping,
        )
        return display_key, random_tensors[tensor_key]

    _, random_summary = _run_condition(
        random_id, conditions_dir / f"{random_id}.jsonl", prompts, controller, evaluator,
        model, tokenizer, max_input_tokens, max_new_tokens, layer, alpha,
        random_vector,
        families, progress_path,
    )
    condition_artifacts[random_id] = random_summary["artifact"]
    specificity = specificity_result(
        float(base_summary["macro_family_accuracy"]),
        float(chosen["macro_family_accuracy"]),
        float(wrong_summary["macro_family_accuracy"]),
        float(random_summary["macro_family_accuracy"]),
    )
    manifest = {
        "type": "lollanet_cma_development_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen" if specificity["passed"] else charter["specificity_gate"]["failure_status"],
        "model_id": direction_manifest["model_id"],
        "revision": direction_manifest["revision"],
        "selected_layer": layer,
        "selected_alpha": alpha,
        "base_summary": base_summary,
        "selected_correct_summary": chosen,
        "wrong_summary": wrong_summary,
        "random_summary": random_summary,
        "specificity": specificity,
        "hard_gates": {
            "all_condition_coverage_complete": all(
                int(summary["record_count"]) == int(charter["data"]["expected_records"])
                for summary in [base_summary, *grid_summaries, wrong_summary, random_summary]
            ),
            "selection_frozen_before_controls": selection_path.is_file(),
            "direction_specificity": bool(specificity["passed"]),
        },
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "access_view_manifest_sha256": sha256_file(access_manifest_path),
        "direction_manifest_sha256": sha256_file(direction_manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
        "model_placement": placement,
        "artifacts": {
            "run_intent": {"path": str(intent_path), "sha256": sha256_file(intent_path)},
            "access_view_manifest": {"path": str(access_manifest_path.resolve()), "sha256": sha256_file(access_manifest_path)},
            "direction_manifest": {"path": str(direction_manifest_path.resolve()), "sha256": sha256_file(direction_manifest_path)},
            "preflight": {"path": str(preflight_path.resolve()), "sha256": sha256_file(preflight_path)},
            "grid_selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "random_vectors": {"path": str(random_path), "sha256": sha256_file(random_path)},
            "conditions": condition_artifacts,
        },
    }
    manifest["hard_gate_pass"] = all(manifest["hard_gates"].values())
    manifest["development_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_json(progress_path, {
        "type": "lollanet_cma_development_progress_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "complete",
        "completed": len(condition_artifacts),
        "total": 1 + len(layers) * len(alphas) + 2,
    })
    return manifest
