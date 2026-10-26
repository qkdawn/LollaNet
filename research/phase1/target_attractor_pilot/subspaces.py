from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers


@dataclass(frozen=True)
class SubspaceTarget:
    basis: np.ndarray
    target_coordinates: np.ndarray
    rank: int
    explained_variance: float


def fit_subspace_target(
    differences: np.ndarray,
    positive: np.ndarray,
    variance_threshold: float = 0.80,
    maximum_rank: int = 8,
) -> SubspaceTarget:
    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("subspace fitting requires a matrix with at least two contrasts")
    _left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    variance = singular ** 2
    if not np.isfinite(variance).all() or float(variance.sum()) <= 0:
        raise ValueError("factor contrasts are degenerate")
    cumulative = np.cumsum(variance) / variance.sum()
    rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    rank = max(1, min(rank, maximum_rank, right.shape[0]))
    basis = right[:rank].T.astype(np.float32)
    target_coordinates = (np.asarray(positive, dtype=np.float32) @ basis).mean(axis=0)
    return SubspaceTarget(
        basis=basis,
        target_coordinates=target_coordinates.astype(np.float32),
        rank=rank,
        explained_variance=float(cumulative[rank - 1]),
    )


def principal_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = np.linalg.svd(np.asarray(left).T @ np.asarray(right), compute_uv=False)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def target_distance(matrix: np.ndarray, target: SubspaceTarget) -> np.ndarray:
    coordinates = np.asarray(matrix) @ target.basis
    return np.linalg.norm(coordinates - target.target_coordinates, axis=1)


def combined_correction(
    hidden: np.ndarray,
    targets: list[SubspaceTarget],
    gain: float,
    maximum_norm: float,
) -> np.ndarray:
    value = np.asarray(hidden, dtype=np.float32)
    correction = np.zeros_like(value)
    for target in targets:
        coordinates = value @ target.basis
        correction += (target.target_coordinates - coordinates) @ target.basis.T
    correction *= float(gain) / max(len(targets), 1)
    norm = float(np.linalg.norm(correction))
    if norm > maximum_norm > 0:
        correction *= float(maximum_norm) / norm
    return correction.astype(np.float32)


def norm_matched_random_subspace(target: SubspaceTarget, hidden_size: int, seed: int) -> SubspaceTarget:
    rng = np.random.default_rng(seed)
    random = rng.standard_normal((hidden_size, target.rank))
    basis, _ = np.linalg.qr(random)
    basis = basis[:, :target.rank].astype(np.float32)
    radius = float(np.linalg.norm(target.target_coordinates))
    coordinates = rng.standard_normal(target.rank).astype(np.float32)
    coordinates *= radius / max(float(np.linalg.norm(coordinates)), 1e-12)
    return SubspaceTarget(basis, coordinates, target.rank, target.explained_variance)


@contextmanager
def attract_to_subspaces(
    model: Any,
    layer_index: int,
    targets_by_window: dict[str, list[SubspaceTarget]],
    gain: float,
    maximum_norm: float,
) -> Iterator[dict[str, Any]]:
    import torch

    state: dict[str, Any] = {"enabled": False, "window": "early_4", "calls": 0, "delta_norms": []}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if not state["enabled"]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        current = hidden[:, -1, :].detach().float().cpu().numpy()[0]
        correction = combined_correction(
            current, targets_by_window[str(state["window"])], gain, maximum_norm,
        )
        delta = torch.as_tensor(correction, device=hidden.device, dtype=hidden.dtype)
        modified = hidden.clone()
        modified[:, -1, :] += delta
        state["calls"] += 1
        state["delta_norms"].append(float(np.linalg.norm(correction)))
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = _layers(model)[layer_index].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()
