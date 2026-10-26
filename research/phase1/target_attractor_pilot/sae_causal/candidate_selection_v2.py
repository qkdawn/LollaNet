from __future__ import annotations

from typing import Any

import numpy as np

from .candidates import FeatureCandidate, _column, _rho
from .contracts import AXES
from .sae import SparseCodes


def rank_candidates_train_only(
    codes: SparseCodes, residuals: Any, candidates_per_axis: int = 24,
    minimum_active_train: int = 12,
) -> dict[str, list[FeatureCandidate]]:
    """Freeze candidates using train labels; development is report-only."""
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
            rows.append(FeatureCandidate(
                axis=axis,
                feature=int(feature),
                train_rho=_rho(values[train], residuals.residual[axis][train]),
                development_rho=_rho(values[development], residuals.residual[axis][development]),
                train_active=int((values[train] != 0).sum()),
                development_active=int((values[development] != 0).sum()),
            ))
        rows.sort(key=lambda row: abs(row.train_rho), reverse=True)
        output[axis] = rows[:candidates_per_axis]
    return output
