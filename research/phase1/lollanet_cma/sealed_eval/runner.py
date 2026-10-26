from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..directions.preflight import _verify_checkpoint_files
from ..evaluation.preflight import _verified_content, verify_development_run
from ..evaluation.release import validate_released_view
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..qualification.runner import SYSTEM_INSTRUCTION, _model_placement
from .conditions import (
    causal_cot_prompt,
    validate_condition_contract,
    validate_general_equal_all,
    vector_reference,
)
from .control_set import validate_control_bundle
from .generation import (
    completion_mean_log_probability,
    generate_paired_steered,
    generate_unsteered,
    parse_option_letter,
)
from .metrics import binary_brier, summarize_rows, yes_probability
from .prepare_probe import _array_hash


YES_COMPLETION = "Final answer: yes"
NO_COMPLETION = "Final answer: no"


def _verified_artifact(source: dict[str, Any], label: str) -> Path:
    path = Path(str(source.get("path", "")))
    if not path.is_file() or sha256_file(path) != source.get("sha256"):
        raise ValueError(f"{label} is missing or changed")
    return path


def _render(tokenizer: Any, system: str, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_prefix(path: Path, expected_ids: list[str], condition_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids[:len(rows)] or len(observed) != len(set(observed)):
        raise ValueError(f"sealed condition is not an exact ordered prefix: {condition_id}")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content) or row.get("condition_id") != condition_id:
            raise ValueError(f"sealed condition row changed: {condition_id} {row.get('record_id')}")
    return rows


def _open_activations(
    path: Path, record_count: int, hidden_size: int, rows: list[dict[str, Any]],
) -> np.ndarray:
    if path.is_file():
        values = np.load(path, mmap_mode="r+")
        if values.shape != (record_count, hidden_size) or values.dtype != np.float32:
            raise ValueError("sealed activation matrix has a different shape or dtype")
    else:
        values = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(record_count, hidden_size),
        )
    for index, row in enumerate(rows):
        if row.get("activation_sha256") != _array_hash(values[index]):
            raise ValueError(f"sealed activation row changed: {row.get('record_id')}")
    return values


def _random_vectors(
    directions: dict[str, Any],
    families: list[str],
    layer: int,
    seeds: dict[str, int],
) -> dict[str, np.ndarray]:
    hidden_size = int(directions[f"{families[0]}__layer_{layer}"].numel())
    result: dict[str, np.ndarray] = {}
    for condition_id, seed in seeds.items():
        rng = np.random.default_rng(int(seed))
        for family in families:
            reference = directions[f"{family}__layer_{layer}"].detach().cpu().float().numpy()
            reference_norm = float(np.linalg.norm(reference))
            value = rng.standard_normal(hidden_size).astype(np.float64)
            norm = float(np.linalg.norm(value))
            if not np.isfinite(norm) or norm <= 0 or not np.isfinite(reference_norm) or reference_norm <= 0:
                raise ValueError("sealed random direction is non-finite or zero")
            result[f"{condition_id}::{family}__layer_{layer}"] = (
                value / norm * reference_norm
            ).astype(np.float32)
    return result


def _freeze_random_vectors(
    path: Path, values: dict[str, np.ndarray],
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    expected = {key: torch.from_numpy(value) for key, value in values.items()}
    if path.is_file():
        observed = load_file(path, device="cpu")
        if set(observed) != set(expected) or any(
            not torch.equal(observed[key].cpu(), expected[key]) for key in expected
        ):
            raise ValueError("sealed random direction artifact differs from its frozen seeds")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(expected, temporary, metadata={"distribution": "normal_norm_matched"})
        temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "tensor_sha256": {key: _array_hash(value) for key, value in sorted(values.items())},
    }


def _vector_for_condition(
    condition_id: str,
    family: str,
    layer: int,
    directions: dict[str, Any],
    random_tensors: dict[str, Any],
    wrong_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> tuple[str | None, Any | None]:
    display_key, reference_key = vector_reference(
        condition_id, family, layer, wrong_mapping, shuffled_mapping,
    )
    if display_key is None:
        return None, None
    if condition_id.startswith("random_norm_matched__draw_"):
        return display_key, random_tensors[display_key]
    return display_key, directions[str(reference_key)]


def _load_view(view_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validate_released_view(view_dir, split)
    manifest = read_json(view_dir / "manifest.json")
    prompts = read_jsonl(Path(manifest["artifacts"]["prompts"]["path"]))
    controllers = read_jsonl(Path(manifest["artifacts"]["controller"]["path"]))
    evaluators = read_jsonl(Path(manifest["artifacts"]["evaluator"]["path"]))
    ids = [str(row["record_id"]) for row in prompts]
    if ids != [str(row["record_id"]) for row in controllers] or ids != [str(row["record_id"]) for row in evaluators]:
        raise ValueError(f"released view ordered identities differ: {split}")
    return prompts, {str(row["record_id"]): row for row in controllers}, {
        str(row["record_id"]): row for row in evaluators
    }


def _run_causal_condition(
    condition_id: str,
    output_path: Path,
    activation_path: Path | None,
    prompts: list[dict[str, Any]],
    controller: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
    charter: dict[str, Any],
    directions: dict[str, Any],
    random_tensors: dict[str, Any],
    layer: int,
    alpha: float,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = [str(row["record_id"]) for row in prompts]
    rows = _load_prefix(output_path, expected_ids, condition_id)
    hidden_size = int(model.config.hidden_size)
    activations = None if activation_path is None else _open_activations(
        activation_path, len(prompts), hidden_size, rows,
    )
    maximum_input_tokens = int(charter["execution"]["maximum_input_tokens"])
    maximum_new_tokens = int(charter["execution"]["maximum_new_tokens"])
    template_path = Path(__file__).with_name(str(charter["prompt_only"]["template"]))
    for index in range(len(rows), len(prompts)):
        prompt_row = prompts[index]
        record_id = expected_ids[index]
        metadata = controller[record_id]
        family = str(metadata["model_family"])
        prompt = str(prompt_row["prompt"])
        if condition_id == "causal_cot":
            prompt = causal_cot_prompt(prompt, template_path)
        rendered = _render(tokenizer, SYSTEM_INSTRUCTION, prompt)
        vector_key, vector = _vector_for_condition(
            condition_id, family, layer, directions, random_tensors,
            charter["conditions"]["wrong_mapping"], charter["conditions"]["shuffled_mapping"],
        )
        if vector is None:
            generated = generate_unsteered(
                model, tokenizer, rendered, maximum_input_tokens, maximum_new_tokens, layer,
            )
            score_layer = score_vector = score_alpha = None
        else:
            generated = generate_paired_steered(
                model, tokenizer, rendered, maximum_input_tokens, maximum_new_tokens,
                layer, vector, alpha,
            )
            score_layer, score_vector, score_alpha = layer, vector, alpha
        yes_score = completion_mean_log_probability(
            model, tokenizer, rendered, YES_COMPLETION, maximum_input_tokens,
            score_layer, score_vector, score_alpha,
        )
        no_score = completion_mean_log_probability(
            model, tokenizer, rendered, NO_COMPLETION, maximum_input_tokens,
            score_layer, score_vector, score_alpha,
        )
        gold = str(evaluator[record_id]["answer"]).lower()
        yes_value = yes_probability(yes_score, no_score)
        parsed = generated.get("parsed_answer")
        confidence = (
            yes_value if parsed == "yes" else 1.0 - yes_value if parsed == "no"
            else max(yes_value, 1.0 - yes_value)
        )
        activation = np.asarray(generated.pop("activation"), dtype=np.float32)
        activation_sha256 = None
        if activations is not None:
            activations[index] = activation
            activations.flush()
            activation_sha256 = _array_hash(activation)
        row = {
            "condition_id": condition_id,
            "record_id": record_id,
            "model_family": family,
            "query_type": str(metadata["query_type"]),
            "rung": int(metadata["rung"]),
            "gold_answer": gold,
            "layer": None if vector is None else layer,
            "alpha": None if vector is None else alpha,
            "vector_key": vector_key,
            "activation_sha256": activation_sha256,
            "yes_mean_log_probability": yes_score,
            "no_mean_log_probability": no_score,
            "gold_answer_margin": yes_score - no_score if gold == "yes" else no_score - yes_score,
            "yes_probability": yes_value,
            "predicted_answer_probability": confidence,
            "brier_score": binary_brier(yes_value, gold),
            "correct": bool(generated["parse_ok"] and parsed == gold),
            **generated,
        }
        row["record_content_sha256"] = stable_hash(row)
        rows.append(row)
        atomic_write_jsonl(output_path, rows)
        atomic_write_json(progress_path, {
            "type": "lollanet_cma_sealed_evaluation_progress_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "causal_condition", "condition_id": condition_id,
            "completed": len(rows), "total": len(prompts),
        })
    families = list(charter["families"])
    summary = summarize_rows(rows, families, int(charter["metrics"]["ece_bins"]))
    artifacts: dict[str, Any] = {
        "rows": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
    }
    if activation_path is not None:
        artifacts["activations"] = {
            "path": str(activation_path.resolve()), "sha256": sha256_file(activation_path),
        }
    return rows, {"condition_id": condition_id, **summary, "artifacts": artifacts}


def _control_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(bool(row["correct"]))
    return {
        "record_count": len(rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "category_accuracy": {
            category: float(np.mean(values)) for category, values in sorted(grouped.items())
        },
        "parse_failure_rate": float(np.mean([not bool(row["parse_ok"]) for row in rows])),
    }


def _run_control_condition(
    condition_id: str,
    output_path: Path,
    prompts: list[dict[str, Any]],
    controller: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
    sealed_charter: dict[str, Any],
    control_charter: dict[str, Any],
    directions: dict[str, Any],
    layer: int,
    alpha: float,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = [str(row["record_id"]) for row in prompts]
    rows = _load_prefix(output_path, expected_ids, condition_id)
    maximum_input_tokens = int(sealed_charter["execution"]["maximum_input_tokens"])
    maximum_new_tokens = int(sealed_charter["execution"]["maximum_new_tokens"])
    system = str(control_charter["prompt"]["system_instruction"])
    for index in range(len(rows), len(prompts)):
        prompt_row = prompts[index]
        record_id = expected_ids[index]
        rendered = _render(tokenizer, system, str(prompt_row["prompt"]))
        if condition_id == "base":
            vector_key, vector = None, None
            generated = generate_unsteered(
                model, tokenizer, rendered, maximum_input_tokens, maximum_new_tokens,
                layer, parser=parse_option_letter,
            )
        else:
            family = condition_id.removeprefix("family_direction__")
            vector_key = f"{family}__layer_{layer}"
            vector = directions[vector_key]
            generated = generate_paired_steered(
                model, tokenizer, rendered, maximum_input_tokens, maximum_new_tokens,
                layer, vector, alpha, parser=parse_option_letter,
            )
        generated.pop("activation")
        gold = str(evaluator[record_id]["answer"]).upper()
        row = {
            "condition_id": condition_id,
            "record_id": record_id,
            "category": str(controller[record_id]["category"]),
            "gold_answer": gold,
            "layer": None if vector is None else layer,
            "alpha": None if vector is None else alpha,
            "vector_key": vector_key,
            "correct": bool(generated["parse_ok"] and generated["parsed_answer"] == gold),
            **generated,
        }
        row["record_content_sha256"] = stable_hash(row)
        rows.append(row)
        atomic_write_jsonl(output_path, rows)
        atomic_write_json(progress_path, {
            "type": "lollanet_cma_sealed_evaluation_progress_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "control_condition", "condition_id": condition_id,
            "completed": len(rows), "total": len(prompts),
        })
    return rows, {
        "condition_id": condition_id,
        **_control_summary(rows),
        "artifact": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
    }


def _control_rows(bundle_dir: Path, control_charter: dict[str, Any]) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
]:
    validate_control_bundle(bundle_dir, control_charter)
    manifest = read_json(bundle_dir / "manifest.json")
    prompts = read_jsonl(Path(manifest["artifacts"]["prompts"]["path"]))
    controllers = read_jsonl(Path(manifest["artifacts"]["controller"]["path"]))
    evaluators = read_jsonl(Path(manifest["artifacts"]["evaluator"]["path"]))
    return prompts, {str(row["record_id"]): row for row in controllers}, {
        str(row["record_id"]): row for row in evaluators
    }


def run_sealed_evaluation(
    charter_path: Path,
    control_charter_path: Path,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    control_charter = yaml.safe_load(control_charter_path.read_text(encoding="utf-8"))
    contract = validate_condition_contract(charter)
    preflight = _verified_content(
        read_json(preflight_path), "sealed_evaluation_preflight_content_sha256",
        "sealed evaluation preflight",
    )
    if (
        preflight.get("type") != "lollanet_cma_sealed_evaluation_preflight_v1"
        or preflight.get("sealed_evaluation_gate_status") != "ready"
        or not preflight.get("evaluation_authorized")
        or preflight.get("errors")
        or preflight.get("charter", {}).get("sha256") != sha256_file(charter_path)
        or preflight.get("control_charter", {}).get("sha256") != sha256_file(control_charter_path)
        or preflight.get("code") != runtime_code_identity(Path(__file__).parent)
        or preflight.get("dependency_lock") != dependency_lock()
        or preflight.get("environment") != runtime_environment()
    ):
        raise ValueError("sealed evaluation preflight is not ready for this runtime")
    release_path = Path(preflight["evaluation_release"]["path"])
    probe_manifest_path = Path(preflight["probe"]["path"])
    if (
        sha256_file(release_path) != preflight["evaluation_release"]["sha256"]
        or sha256_file(probe_manifest_path) != preflight["probe"]["sha256"]
    ):
        raise ValueError("sealed release or probe changed after evaluation preflight")
    release = _verified_content(
        read_json(release_path), "evaluation_release_content_sha256", "evaluation release",
    )
    probe_manifest = _verified_content(
        read_json(probe_manifest_path), "probe_content_sha256", "hidden-state probe",
    )
    development_manifest_path = Path(probe_manifest["development_manifest"]["path"])
    if sha256_file(development_manifest_path) != release["development_manifest_sha256"]:
        raise ValueError("probe and evaluation release use different development runs")
    development = verify_development_run(
        development_manifest_path,
        str(charter["prerequisites"]["development_manifest_type"]),
        str(charter["prerequisites"]["development_status"]),
    )
    development_manifest = read_json(development_manifest_path)
    layer = int(development["selected_layer"])
    alpha = float(development["selected_alpha"])
    if layer != int(probe_manifest["selected_layer"]):
        raise ValueError("probe and development selected different layers")
    direction_manifest_path = _verified_artifact(
        development_manifest["artifacts"]["direction_manifest"], "direction manifest",
    )
    direction_manifest = _verified_content(
        read_json(direction_manifest_path), "direction_content_sha256", "direction run",
    )
    directions_path = _verified_artifact(
        direction_manifest["artifacts"]["directions"], "direction tensors",
    )
    directions = load_file(directions_path, device="cpu")
    checkpoint = Path(direction_manifest["checkpoint"])
    checkpoint_identity = _verify_checkpoint_files(checkpoint)
    if checkpoint_identity["checkpoint_content_sha256"] != direction_manifest["checkpoint_manifest_sha256"]:
        raise ValueError("sealed checkpoint differs from direction fitting")
    families = list(charter["families"])
    required_direction_keys = {
        f"{family}__layer_{layer}" for family in families
    } | {f"general__layer_{layer}", f"equal_all__layer_{layer}"}
    if not required_direction_keys.issubset(directions):
        raise ValueError("sealed direction tensors lack a frozen condition")
    direction_validation = validate_general_equal_all(
        directions, families, layer,
        float(charter["prerequisites"]["general_equal_all_max_cosine_similarity"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    random_path = output_dir / "random_vectors.safetensors"
    random_values = _random_vectors(
        directions, families, layer, charter["conditions"]["random_direction_seeds"],
    )
    random_artifact = _freeze_random_vectors(random_path, random_values)
    random_tensors = {key: torch.from_numpy(value) for key, value in random_values.items()}
    intent = {
        "type": "lollanet_cma_sealed_evaluation_intent_v1",
        "charter_sha256": sha256_file(charter_path),
        "control_charter_sha256": sha256_file(control_charter_path),
        "preflight": {"path": str(preflight_path.resolve()), "sha256": sha256_file(preflight_path)},
        "evaluation_release": {"path": str(release_path.resolve()), "sha256": sha256_file(release_path)},
        "probe_manifest": {
            "path": str(probe_manifest_path.resolve()), "sha256": sha256_file(probe_manifest_path),
        },
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "direction_manifest_sha256": sha256_file(direction_manifest_path),
        "direction_tensors_sha256": sha256_file(directions_path),
        "checkpoint_content_sha256": checkpoint_identity["checkpoint_content_sha256"],
        "selected_layer": layer, "selected_alpha": alpha,
        "condition_contract": contract,
        "direction_validation": direction_validation,
        "random_vectors": random_artifact,
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(), "environment": runtime_environment(),
    }
    intent["intent_content_sha256"] = stable_hash(intent)
    intent_path = output_dir / "run_intent.json"
    if intent_path.is_file():
        if read_json(intent_path) != intent:
            raise ValueError("existing sealed evaluation intent differs")
    else:
        atomic_write_json(intent_path, intent)
    set_seed(int(charter["execution"]["seed"]))
    dtype = torch.bfloat16 if direction_manifest["compute_dtype"] == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if stable_hash(tokenizer.chat_template) != direction_manifest["chat_template_sha256"]:
        raise ValueError("sealed tokenizer chat template differs from direction fitting")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, quantization_config=quantization,
        device_map="auto", torch_dtype=dtype,
    )
    model.eval()
    placement = _model_placement(model)
    if not placement["loaded_on_cuda"]:
        raise RuntimeError("sealed model is not loaded entirely on CUDA")
    progress_path = output_dir / "progress.json"
    conditions_root = output_dir / "conditions"
    capture_conditions = {
        "base", charter["conditions"]["random_primary"],
        *[f"family_direction__{family}" for family in families],
    }
    split_results: dict[str, Any] = {}
    for split, view_contract in charter["data"]["views"].items():
        prompts, controllers, evaluators = _load_view(Path(view_contract["path"]), split)
        split_dir = conditions_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, Any] = {}
        for condition_id in contract["model_run_condition_ids"]:
            activation_path = (
                split_dir / f"{condition_id}.activations.npy"
                if condition_id in capture_conditions else None
            )
            _, summary = _run_causal_condition(
                condition_id, split_dir / f"{condition_id}.jsonl", activation_path,
                prompts, controllers, evaluators, model, tokenizer, charter, directions,
                random_tensors, layer, alpha, progress_path,
            )
            summaries[condition_id] = summary
        split_results[split] = {
            "view_manifest_sha256": sha256_file(Path(view_contract["path"]) / "manifest.json"),
            "conditions": summaries,
        }
    control_manifest_path = Path(charter["readiness_blockers"]["unrelated_control_dataset"]["artifact"])
    control_prompts, control_controllers, control_evaluators = _control_rows(
        control_manifest_path.parent, control_charter,
    )
    control_dir = conditions_root / "unrelated_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_summaries: dict[str, Any] = {}
    for condition_id in control_charter["evaluation"]["conditions"]:
        _, summary = _run_control_condition(
            condition_id, control_dir / f"{condition_id}.jsonl", control_prompts,
            control_controllers, control_evaluators, model, tokenizer, charter,
            control_charter, directions, layer, alpha, progress_path,
        )
        control_summaries[condition_id] = summary
    manifest = {
        "type": "lollanet_cma_sealed_evaluation_raw_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_raw_conditions",
        "model_id": direction_manifest["model_id"], "revision": direction_manifest["revision"],
        "selected_layer": layer, "selected_alpha": alpha,
        "model_placement": placement,
        "intent": {"path": str(intent_path.resolve()), "sha256": sha256_file(intent_path)},
        "splits": split_results,
        "unrelated_control": {
            "manifest_sha256": sha256_file(control_manifest_path),
            "conditions": control_summaries,
        },
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(), "environment": runtime_environment(),
    }
    manifest["raw_run_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_json(progress_path, {
        "type": "lollanet_cma_sealed_evaluation_progress_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "complete", "completed": len(split_results) + 1,
        "total": len(charter["data"]["views"]) + 1,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw sealed LollaNet-CMA conditions.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("sealed_eval_charter.yaml"))
    parser.add_argument("--control-charter", type=Path, default=Path(__file__).with_name("control_set_charter.yaml"))
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_sealed_evaluation(args.charter, args.control_charter, args.preflight, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
