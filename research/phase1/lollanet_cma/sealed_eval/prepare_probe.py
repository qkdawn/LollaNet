from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..access_views.views import validate_access_view
from ..common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, stable_hash
from ..directions.hooks import capture_residuals
from ..directions.preflight import _verify_checkpoint_files
from ..evaluation.preflight import verify_development_run
from ..qualification.common import dependency_lock, runtime_code_identity, runtime_environment
from ..qualification.runner import SYSTEM_INSTRUCTION, _model_placement
from .diagnostics import evaluate_linear_probe, select_probe_l2


def _render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verified_existing_probe(
    manifest_path: Path,
    charter_path: Path,
    development_manifest_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    content = dict(manifest)
    claimed = content.pop("probe_content_sha256", None)
    if claimed != stable_hash(content):
        raise ValueError("existing probe manifest content changed")
    if (
        manifest.get("type") != "lollanet_cma_hidden_state_probe_v1"
        or manifest.get("status") != "frozen_before_sealed_evaluation"
        or manifest.get("charter", {}).get("sha256") != sha256_file(charter_path)
        or manifest.get("development_manifest", {}).get("sha256")
        != sha256_file(development_manifest_path)
        or manifest.get("code") != runtime_code_identity(Path(__file__).parent)
        or manifest.get("dependency_lock") != dependency_lock()
        or manifest.get("environment") != runtime_environment()
    ):
        raise ValueError("existing probe manifest differs from the current frozen contract")
    artifacts = manifest.get("artifacts", {})
    sources = [artifacts.get("probe", {})]
    for split in ("vector_train", "development"):
        view = Path(str(artifacts.get(split, {}).get("view_path", "")))
        if (
            not view.is_dir()
            or sha256_file(view / "manifest.json")
            != artifacts.get(split, {}).get("view_manifest_sha256")
            or validate_access_view(view, split) != artifacts.get(split, {}).get("validation")
        ):
            raise ValueError(f"existing probe {split} access view is missing or changed")
        sources.extend([
            artifacts.get(split, {}).get("activations", {}),
            artifacts.get(split, {}).get("records", {}),
        ])
    for source in sources:
        path = Path(str(source.get("path", "")))
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise ValueError("existing probe artifact is missing or changed")
    return manifest


def _load_prefix(path: Path, expected_ids: list[str], split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    observed = [str(row.get("record_id")) for row in rows]
    if observed != expected_ids[:len(rows)] or len(observed) != len(set(observed)):
        raise ValueError(f"probe {split} records are not an exact ordered prefix")
    for row in rows:
        content = dict(row)
        claimed = content.pop("record_content_sha256", None)
        if claimed != stable_hash(content) or row.get("split") != split:
            raise ValueError(f"probe {split} record changed: {row.get('record_id')}")
    return rows


def _capture_split(
    split: str,
    view_dir: Path,
    model: Any,
    tokenizer: Any,
    layer: int,
    maximum_input_tokens: int,
    output_dir: Path,
    progress_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validation = validate_access_view(view_dir, split)
    manifest = read_json(view_dir / "manifest.json")
    prompts = read_jsonl(Path(manifest["artifacts"]["prompts"]["path"]))
    evaluator = {
        str(row["record_id"]): row
        for row in read_jsonl(Path(manifest["artifacts"]["evaluator"]["path"]))
    }
    expected_ids = [str(row["record_id"]) for row in prompts]
    records_path = output_dir / f"{split}_activation_records.jsonl"
    activations_path = output_dir / f"{split}_activations.npy"
    records = _load_prefix(records_path, expected_ids, split)
    hidden_size = int(model.config.hidden_size)
    if activations_path.is_file():
        activations = np.load(activations_path, mmap_mode="r+")
        if activations.shape != (len(prompts), hidden_size) or activations.dtype != np.float32:
            raise ValueError(f"probe {split} activation matrix has a different shape or dtype")
    else:
        activations = np.lib.format.open_memmap(
            activations_path, mode="w+", dtype=np.float32, shape=(len(prompts), hidden_size),
        )
    for index, record in enumerate(records):
        if record["activation_sha256"] != _array_hash(activations[index]):
            raise ValueError(f"probe {split} activation row changed: {record['record_id']}")
    input_device = model.get_input_embeddings().weight.device
    for index in range(len(records), len(prompts)):
        prompt_row = prompts[index]
        record_id = expected_ids[index]
        rendered = _render(tokenizer, str(prompt_row["prompt"]))
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded["input_ids"].shape[1])
        if input_tokens > maximum_input_tokens:
            raise ValueError(f"probe input exceeds the frozen token limit: {record_id}")
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        import torch
        with torch.inference_mode(), capture_residuals(model, [layer], input_tokens - 1) as captured:
            model(**encoded, use_cache=False)
        value = captured[layer].numpy().astype(np.float32, copy=False)
        activations[index] = value
        activations.flush()
        answer = str(evaluator[record_id]["answer"]).lower()
        if answer not in {"yes", "no"}:
            raise ValueError(f"probe evaluator label is not binary: {record_id}")
        row = {
            "split": split,
            "record_id": record_id,
            "label": 1 if answer == "yes" else 0,
            "input_tokens": input_tokens,
            "activation_sha256": _array_hash(value),
        }
        row["record_content_sha256"] = stable_hash(row)
        records.append(row)
        atomic_write_jsonl(records_path, records)
        atomic_write_json(progress_path, {
            "type": "lollanet_cma_probe_progress_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": split,
            "completed": len(records),
            "total": len(prompts),
        })
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    return np.asarray(activations), labels, {
        "view_path": str(view_dir.resolve()),
        "view_manifest_sha256": sha256_file(view_dir / "manifest.json"),
        "validation": validation,
        "activations": {"path": str(activations_path.resolve()), "sha256": sha256_file(activations_path)},
        "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
    }


def prepare_probe(
    charter_path: Path,
    development_manifest_path: Path,
    vector_train_view: Path,
    development_view: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    contract = charter["hidden_state_probe"]
    development = verify_development_run(
        development_manifest_path,
        str(charter["prerequisites"]["development_manifest_type"]),
        str(charter["prerequisites"]["development_status"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        return _verified_existing_probe(manifest_path, charter_path, development_manifest_path)
    development_manifest = read_json(development_manifest_path)
    direction_manifest_path = Path(development_manifest["artifacts"]["direction_manifest"]["path"])
    direction_manifest = read_json(direction_manifest_path)
    checkpoint = Path(direction_manifest["checkpoint"])
    checkpoint_identity = _verify_checkpoint_files(checkpoint)
    if checkpoint_identity["checkpoint_content_sha256"] != direction_manifest["checkpoint_manifest_sha256"]:
        raise ValueError("probe checkpoint differs from the direction run")
    progress_path = output_dir / "progress.json"
    layer = int(development["selected_layer"])
    set_seed(int(charter["execution"]["seed"]))
    dtype = torch.bfloat16 if direction_manifest["compute_dtype"] == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if stable_hash(tokenizer.chat_template) != direction_manifest["chat_template_sha256"]:
        raise ValueError("probe tokenizer chat template differs from direction fitting")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, quantization_config=quantization,
        device_map="auto", torch_dtype=dtype,
    )
    model.eval()
    placement = _model_placement(model)
    if not placement["loaded_on_cuda"]:
        raise RuntimeError("probe model is not loaded entirely on CUDA")
    maximum_input_tokens = int(charter["execution"]["maximum_input_tokens"])
    train_x, train_y, train_artifacts = _capture_split(
        "vector_train", vector_train_view, model, tokenizer, layer, maximum_input_tokens,
        output_dir, progress_path,
    )
    development_x, development_y, development_artifacts = _capture_split(
        "development", development_view, model, tokenizer, layer, maximum_input_tokens,
        output_dir, progress_path,
    )
    probe, grid = select_probe_l2(
        train_x, train_y, development_x, development_y,
        [float(value) for value in contract["l2_grid"]],
        float(contract["standard_deviation_floor"]),
    )
    probe_path = output_dir / "probe.npz"
    temporary = output_dir / "probe.tmp.npz"
    np.savez(
        temporary,
        weights=probe["weights"], feature_mean=probe["feature_mean"],
        feature_scale=probe["feature_scale"], intercept=np.asarray([probe["intercept"]]),
        l2=np.asarray([probe["l2"]]),
    )
    temporary.replace(probe_path)
    selected_metrics = evaluate_linear_probe(probe, development_x, development_y)
    manifest = {
        "type": "lollanet_cma_hidden_state_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_sealed_evaluation",
        "model_id": development_manifest["model_id"],
        "revision": development_manifest["revision"],
        "selected_layer": layer,
        "selected_l2": float(probe["l2"]),
        "development_metrics": selected_metrics,
        "grid_summaries": grid,
        "charter": {"path": str(charter_path.resolve()), "sha256": sha256_file(charter_path)},
        "development_manifest": {
            "path": str(development_manifest_path.resolve()), "sha256": sha256_file(development_manifest_path),
        },
        "model_placement": placement,
        "artifacts": {
            "vector_train": train_artifacts,
            "development": development_artifacts,
            "probe": {"path": str(probe_path.resolve()), "sha256": sha256_file(probe_path)},
        },
        "code": runtime_code_identity(Path(__file__).parent),
        "dependency_lock": dependency_lock(),
        "environment": runtime_environment(),
    }
    manifest["probe_content_sha256"] = stable_hash(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(progress_path, {
        "type": "lollanet_cma_probe_progress_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "complete", "completed": 2, "total": 2,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and freeze the LollaNet-CMA hidden-state probe.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("sealed_eval_charter.yaml"))
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--vector-train-view", type=Path, required=True)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_probe(
        args.charter, args.development_manifest, args.vector_train_view,
        args.development_view, args.output,
    )
    print(result["status"])


if __name__ == "__main__":
    main()
