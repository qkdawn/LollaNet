from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import atomic_write_json, read_jsonl, sha256_file, stable_hash
from .runtime_audit import audit_run_manifest

FACTORS = ("reward", "social", "certainty")


def factor_key(row: dict[str, Any]) -> str | None:
    if row.get("control_type") != "factorial":
        return None
    active = [
        name for name, field in (("reward", "reward_lure"), ("social", "social_lure"), ("certainty", "certainty_pressure"))
        if int(row.get(field, 0)) == 1
    ]
    return active[0] if len(active) == 1 else ("neutral" if not active else None)


def _activation_key(record_id: str, layer_index: int) -> str:
    return f"{record_id}__layer_{layer_index}"


def fit_directions(
    prompts_path: Path,
    activations_path: Path,
    run_manifest_path: Path,
    output_path: Path,
    ranks: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
    if output_path.suffix.lower() != ".npz":
        raise ValueError("direction output path must end in .npz")
    all_prompts = read_jsonl(prompts_path)
    record_ids = [str(row["record_id"]) for row in all_prompts]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("prompt file contains duplicate record_id")
    prompts = [row for row in all_prompts if row["split"] == "discovery"]
    if not prompts:
        raise ValueError("prompt file contains no discovery records")
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prompts:
        key = factor_key(row)
        if key is not None:
            item_id = str(row["base_item_id"])
            if key in by_item[item_id]:
                raise ValueError(f"duplicate discovery condition {key!r} for {item_id}")
            by_item[item_id][key] = row
    required_conditions = {"neutral", *FACTORS}
    incomplete = {
        item_id: sorted(required_conditions - set(conditions))
        for item_id, conditions in by_item.items()
        if set(conditions) != required_conditions
    }
    if incomplete:
        raise ValueError(f"incomplete discovery condition coverage: {incomplete}")
    import json
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_audit = audit_run_manifest(run_manifest_path)
    if run_audit.get("status") != "current_complete" or not run_audit.get("direction_fit_eligible"):
        raise ValueError(f"direction fitting requires a current complete audited baseline: {run_audit}")
    if run_manifest.get("method") != "baseline":
        raise ValueError("direction fitting requires a baseline run manifest")
    if run_manifest.get("requested_split") != "all":
        raise ValueError("direction fitting requires a complete all-split baseline run")
    if run_manifest.get("activation_capture_policy") != "last_prompt_token_pre_intervention":
        raise ValueError("baseline activation capture policy is incompatible with direction fitting")
    if run_manifest.get("inputs", {}).get("sha256") != sha256_file(prompts_path):
        raise ValueError("run manifest input hash does not match prompt file")
    if int(run_manifest.get("record_count", -1)) != len(all_prompts):
        raise ValueError("baseline run is incomplete; limited runs cannot fit formal directions")
    if run_manifest.get("record_ids_sha256") != stable_hash(sorted(record_ids)):
        raise ValueError("baseline run record ID hash does not match prompts")
    if run_manifest.get("artifacts", {}).get("activations", {}).get("sha256") != sha256_file(activations_path):
        raise ValueError("run manifest activation hash does not match activation archive")
    layer_indices = [int(value) for value in run_manifest["layer_indices"]]
    if not layer_indices or len(layer_indices) != len(set(layer_indices)):
        raise ValueError("run manifest layer_indices must be unique and non-empty")
    ranks = tuple(int(rank) for rank in ranks)
    if not ranks or any(rank < 1 for rank in ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("ranks must contain unique positive integers")
    archive = np.load(activations_path)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "type": "lollasteer_q_direction_bundle", "source_run": run_manifest.get("run_id"),
        "source_run_manifest": {
            "path": str(run_manifest_path.resolve()), "sha256": sha256_file(run_manifest_path),
        },
        "source_activations": {
            "path": str(activations_path.resolve()), "sha256": sha256_file(activations_path),
        },
        "layer_indices": layer_indices, "ranks": list(ranks), "factors": {},
        "checkpoint_actual": run_manifest.get("checkpoint_actual"),
        "checkpoint_revision": run_manifest.get("checkpoint_revision"),
        "chat_template_sha256": run_manifest.get("chat_template_sha256"),
        "dtype": run_manifest.get("dtype"),
        "dependency_lock_sha256": run_manifest.get("dependency_lock", {}).get("sha256"),
        "runtime_code_sha256": run_manifest.get("runtime_code", {}).get("sha256"),
        "runtime_environment_sha256": run_manifest.get("runtime_environment_sha256"),
        "prompt_bundle_sha256": run_manifest.get("inputs", {}).get("sha256"),
    }
    for factor in FACTORS:
        metadata["factors"][factor] = {}
        for layer_index in layer_indices:
            differences: list[np.ndarray] = []
            neutral_values: list[np.ndarray] = []
            pair_ids: list[str] = []
            for item_id, conditions in sorted(by_item.items()):
                if "neutral" not in conditions or factor not in conditions:
                    continue
                neutral_id = conditions["neutral"]["record_id"]
                harmful_id = conditions[factor]["record_id"]
                neutral_key = _activation_key(neutral_id, layer_index)
                harmful_key = _activation_key(harmful_id, layer_index)
                if neutral_key not in archive or harmful_key not in archive:
                    raise ValueError(
                        f"missing discovery activation for {item_id}/{factor}/layer_{layer_index}"
                    )
                neutral = np.asarray(archive[neutral_key], dtype=np.float64)
                harmful = np.asarray(archive[harmful_key], dtype=np.float64)
                if neutral.ndim != 1 or harmful.ndim != 1 or neutral.shape != harmful.shape:
                    raise ValueError(f"activation shape mismatch for {item_id}/{factor}/layer_{layer_index}")
                if not np.all(np.isfinite(neutral)) or not np.all(np.isfinite(harmful)):
                    raise ValueError(f"non-finite activation for {item_id}/{factor}/layer_{layer_index}")
                if differences and neutral.shape != differences[0].shape:
                    raise ValueError(f"inconsistent hidden size for {factor}/layer_{layer_index}")
                differences.append(harmful - neutral)
                neutral_values.append(neutral)
                pair_ids.append(item_id)
            if len(pair_ids) != len(by_item):
                raise ValueError(f"incomplete discovery pairs for {factor} layer {layer_index}")
            matrix = np.stack(differences)
            caa = matrix.mean(axis=0)
            norm = float(np.linalg.norm(caa))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"degenerate CAA direction for {factor} layer {layer_index}")
            caa = caa / norm
            _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
            numerical_rank = int(np.linalg.matrix_rank(matrix))
            max_rank = min(max(ranks), numerical_rank, right.shape[0])
            if max_rank < 1:
                raise ValueError(f"rank-zero difference matrix for {factor} layer {layer_index}")
            basis = right[:max_rank]
            key_prefix = f"{factor}__layer_{layer_index}"
            arrays[f"{key_prefix}__caa"] = caa.astype(np.float32)
            arrays[f"{key_prefix}__basis"] = basis.astype(np.float32)
            arrays[f"{key_prefix}__neutral_center"] = np.stack(neutral_values).mean(axis=0).astype(np.float32)
            metadata["factors"][factor][str(layer_index)] = {
                "pairs": len(pair_ids), "pair_ids_sha256": stable_hash(pair_ids),
                "singular_values": singular_values[:max_rank].tolist(), "caa_original_l2": norm,
                "available_rank": max_rank, "numerical_rank": numerical_rank,
                "hidden_size": int(matrix.shape[1]),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    metadata["npz_path"] = str(output_path)
    metadata["npz_sha256"] = sha256_file(output_path)
    metadata_path = output_path.with_suffix(".json")
    atomic_write_json(metadata_path, metadata)
    return metadata


class Intervention:
    def __init__(self, archive_path: Path, method: str, factor: str, layer_index: int, strength: float, rank: int = 1):
        if method not in {"caa", "low_rank"}:
            raise ValueError("method must be caa or low_rank")
        if factor not in FACTORS:
            raise ValueError(f"unknown factor {factor}")
        archive = np.load(archive_path)
        prefix = f"{factor}__layer_{layer_index}"
        self.method = method
        self.factor = factor
        self.layer_index = int(layer_index)
        self.strength = float(strength)
        self.rank = int(rank)
        self.direction = np.asarray(archive[f"{prefix}__caa"], dtype=np.float32)
        self.basis = np.asarray(archive[f"{prefix}__basis"], dtype=np.float32)[: self.rank]
        self.center = np.asarray(archive[f"{prefix}__neutral_center"], dtype=np.float32)
        if not np.isfinite(self.strength):
            raise ValueError("strength must be finite")
        if self.direction.ndim != 1 or self.center.shape != self.direction.shape:
            raise ValueError("direction and center must be aligned one-dimensional vectors")
        if self.basis.ndim != 2 or self.basis.shape[1:] != self.direction.shape:
            raise ValueError("basis hidden size does not match direction")
        if not all(np.all(np.isfinite(value)) for value in (self.direction, self.basis, self.center)):
            raise ValueError("direction bundle contains non-finite values")
        direction_norm = float(np.linalg.norm(self.direction))
        if not np.isclose(direction_norm, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("CAA direction must have unit norm")
        gram = self.basis @ self.basis.T
        if not np.allclose(gram, np.eye(len(self.basis), dtype=np.float32), rtol=1e-4, atol=1e-5):
            raise ValueError("low-rank basis must be row-orthonormal")
        if self.rank < 1 or self.rank > len(self.basis):
            raise ValueError("rank outside fitted basis")

    def delta(self, activation: np.ndarray) -> np.ndarray:
        value = np.asarray(activation, dtype=np.float32)
        if value.shape != self.direction.shape or not np.all(np.isfinite(value)):
            raise ValueError("activation must be a finite vector matching the direction hidden size")
        if self.method == "caa":
            return -self.strength * self.direction
        centered = value - self.center
        projection = self.basis.T @ (self.basis @ centered)
        norm = float(np.linalg.norm(projection))
        if not np.isfinite(norm):
            raise ValueError("low-rank projection norm is non-finite")
        if norm <= np.finfo(np.float32).eps:
            return np.zeros_like(value)
        return -self.strength * projection / norm
