from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .contracts import AXES
from .sae import SparseCodes


@dataclass(frozen=True)
class FeatureCandidate:
    axis: str
    feature: int
    train_rho: float
    development_rho: float
    train_active: int
    development_active: int


def _column(codes: SparseCodes, feature: int) -> np.ndarray:
    return np.where(codes.indices == feature, codes.values, 0.0).sum(axis=1)


def _rho(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def rank_candidates(
    codes: SparseCodes, residuals: Any, candidates_per_axis: int = 24, minimum_active_train: int = 12,
) -> dict[str, list[FeatureCandidate]]:
    if codes.record_ids != residuals.record_ids:
        raise ValueError("SAE codes and labels use different record order")
    train = residuals.split == "train"
    development = residuals.split == "development"
    active_features, counts = np.unique(codes.indices[train], return_counts=True)
    eligible = active_features[counts >= minimum_active_train]
    output: dict[str, list[FeatureCandidate]] = {}
    for axis in AXES:
        rows = []
        for feature in eligible:
            values = _column(codes, int(feature))
            train_rho = _rho(values[train], residuals.residual[axis][train])
            dev_rho = _rho(values[development], residuals.residual[axis][development])
            rows.append(FeatureCandidate(axis, int(feature), train_rho, dev_rho,
                                         int((values[train] != 0).sum()), int((values[development] != 0).sum())))
        rows.sort(key=lambda row: (abs(row.development_rho), abs(row.train_rho)), reverse=True)
        output[axis] = rows[:candidates_per_axis]
    return output


def matched_random_features(codes: SparseCodes, feature: int, count: int, seed: int = 20260726) -> list[int]:
    frequencies = np.bincount(codes.indices.ravel(), minlength=codes.width)
    target = frequencies[int(feature)]
    tolerance = max(2, int(target * 0.2))
    pool = np.flatnonzero((frequencies >= target - tolerance) & (frequencies <= target + tolerance))
    pool = pool[pool != feature]
    if len(pool) < count:
        pool = np.flatnonzero(frequencies > 0)
        pool = pool[pool != feature]
    rng = np.random.default_rng(seed + int(feature))
    return [int(value) for value in rng.choice(pool, size=count, replace=False)]
