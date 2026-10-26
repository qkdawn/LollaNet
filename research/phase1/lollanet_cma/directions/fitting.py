from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Callable

import numpy as np

from ..common import stable_hash


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return sha256(contiguous.tobytes(order="C")).hexdigest()


def layer_indices(layer_count: int, fractions: list[float]) -> list[int]:
    if layer_count < 1:
        raise ValueError("model must expose at least one transformer layer")
    indices = [round((layer_count - 1) * float(fraction)) for fraction in fractions]
    if any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("layer fractions must lie in [0, 1]")
    if len(indices) != len(set(indices)):
        raise ValueError("frozen layer fractions collapse to duplicate layers")
    return indices


def is_eligible(row: dict[str, Any]) -> bool:
    return bool(
        row.get("positive_parse_ok")
        and row.get("positive_correct")
        and float(row.get("positive_gold_margin", float("-inf")))
        > float(row.get("negative_gold_margin", float("inf")))
        and row.get("hidden_finite")
    )


def balanced_record_ids(
    rows: list[dict[str, Any]], families: list[str], labels: list[str], seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    selected: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {}
    for family in families:
        eligible = [row for row in rows if row["model_family"] == family and is_eligible(row)]
        by_label = {
            label: sorted(
                [row for row in eligible if row["gold_answer"] == label],
                key=lambda row: stable_hash({"seed": seed, "record_id": row["record_id"]}),
            )
            for label in labels
        }
        keep_per_label = min(len(values) for values in by_label.values())
        kept = [row for label in labels for row in by_label[label][:keep_per_label]]
        kept.sort(key=lambda row: stable_hash({"seed": seed, "record_id": row["record_id"]}))
        selected[family] = [str(row["record_id"]) for row in kept]
        diagnostics[family] = {
            "eligible_by_label": {label: len(by_label[label]) for label in labels},
            "balanced_per_label": keep_per_label,
            "balanced_total": len(kept),
            "selected_record_ids_sha256": stable_hash(selected[family]),
        }
    return selected, diagnostics


def build_normalized_directions(
    selected_ids: dict[str, list[str]],
    layer_ids: list[int],
    load_contrast: Callable[[str, int], np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"families": {}, "general": {}, "equal_all": {}}
    for family, record_ids in selected_ids.items():
        if not record_ids:
            raise ValueError(f"cannot fit a direction without records: {family}")
        metadata["families"][family] = {}
        for layer_id in layer_ids:
            contrasts = np.stack([load_contrast(record_id, layer_id) for record_id in record_ids]).astype(np.float64)
            mean = contrasts.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if not np.isfinite(mean).all() or not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"direction mean is non-finite or zero: {family} layer {layer_id}")
            key = f"{family}__layer_{layer_id}"
            vectors[key] = (mean / norm).astype(np.float32)
            metadata["families"][family][str(layer_id)] = {
                "pre_normalization_norm": norm,
                "normalized_norm": float(np.linalg.norm(vectors[key].astype(np.float64))),
                "record_count": len(record_ids),
                "dtype": str(vectors[key].dtype),
                "shape": list(vectors[key].shape),
                "tensor_sha256": _array_sha256(vectors[key]),
            }
    for layer_id in layer_ids:
        pooled = np.stack([
            load_contrast(record_id, layer_id)
            for record_ids in selected_ids.values() for record_id in record_ids
        ]).astype(np.float64)
        mean = pooled.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(mean).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"general direction is non-finite or zero: layer {layer_id}")
        vectors[f"general__layer_{layer_id}"] = (mean / norm).astype(np.float32)
        general = vectors[f"general__layer_{layer_id}"]
        metadata["general"][str(layer_id)] = {
            "pre_normalization_norm": norm,
            "normalized_norm": float(np.linalg.norm(general.astype(np.float64))),
            "dtype": str(general.dtype),
            "shape": list(general.shape),
            "tensor_sha256": _array_sha256(general),
            "record_count": int(pooled.shape[0]),
            "family_record_counts": {
                family: len(record_ids) for family, record_ids in selected_ids.items()
            },
        }
        family_vectors = np.stack([
            vectors[f"{family}__layer_{layer_id}"] for family in selected_ids
        ]).astype(np.float64)
        equal_mean = family_vectors.mean(axis=0)
        equal_norm = float(np.linalg.norm(equal_mean))
        if not np.isfinite(equal_mean).all() or not np.isfinite(equal_norm) or equal_norm <= 0:
            raise ValueError(f"equal-all direction is non-finite or zero: layer {layer_id}")
        equal_key = f"equal_all__layer_{layer_id}"
        vectors[equal_key] = (equal_mean / equal_norm).astype(np.float32)
        equal_all = vectors[equal_key]
        cosine = float(np.dot(
            general.astype(np.float64), equal_all.astype(np.float64),
        ) / (
            np.linalg.norm(general.astype(np.float64))
            * np.linalg.norm(equal_all.astype(np.float64))
        ))
        metadata["equal_all"][str(layer_id)] = {
            "pre_normalization_norm": equal_norm,
            "normalized_norm": float(np.linalg.norm(equal_all.astype(np.float64))),
            "dtype": str(equal_all.dtype),
            "shape": list(equal_all.shape),
            "tensor_sha256": _array_sha256(equal_all),
            "source_family_keys": [
                f"{family}__layer_{layer_id}" for family in selected_ids
            ],
            "cosine_similarity_with_general": cosine,
        }
    return vectors, metadata


def eligibility_counts(rows: list[dict[str, Any]], families: list[str]) -> dict[str, int]:
    counts = Counter(row["model_family"] for row in rows if is_eligible(row))
    return {family: counts[family] for family in families}
