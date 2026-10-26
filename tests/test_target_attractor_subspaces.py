from __future__ import annotations

import numpy as np
import pytest

from research.phase1.target_attractor_pilot.factorial import AXES, factorial_rows, matched_factor_contrasts
from research.phase1.target_attractor_pilot.subspaces import (
    SubspaceTarget, combined_correction, fit_subspace_target, principal_cosines,
)


def test_factorial_rows_are_case_matched_and_evaluation_free() -> None:
    rows = factorial_rows()
    assert len(rows) == 20 * 8
    assert {row["split"] for row in rows} == {"train", "development"}
    for case_id in {row["case_id"] for row in rows}:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        assert len(case_rows) == 8
        assert len({(row["specificity"], row["independence"], row["non_template"]) for row in case_rows}) == 8


def test_each_factor_contrast_holds_the_other_two_factors_fixed() -> None:
    rows = factorial_rows(("train",))
    activations = np.asarray([
        [row[axis] for axis in AXES] for row in rows
    ], dtype=np.float32)
    for index, axis in enumerate(AXES):
        contrast = matched_factor_contrasts(rows, activations, axis)["differences"]
        assert contrast.shape[0] == 16 * 4
        expected = np.zeros(3, dtype=np.float32)
        expected[index] = 1.0
        np.testing.assert_allclose(contrast, np.tile(expected, (contrast.shape[0], 1)))


def test_svd_recovers_a_low_rank_factor_subspace() -> None:
    rng = np.random.default_rng(4)
    true_basis = np.asarray([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
    coefficients = rng.normal(size=(40, 2))
    differences = coefficients @ true_basis.T + rng.normal(scale=1e-4, size=(40, 4))
    positive = differences + 2.0
    target = fit_subspace_target(differences, positive, variance_threshold=0.95, maximum_rank=3)
    assert target.rank == 2
    assert principal_cosines(target.basis, true_basis).min() > 0.999


def test_combined_controller_moves_toward_each_subspace_without_unbounded_sum() -> None:
    first = SubspaceTarget(np.asarray([[1.0], [0.0]]), np.asarray([2.0]), 1, 1.0)
    second = SubspaceTarget(np.asarray([[0.0], [1.0]]), np.asarray([-2.0]), 1, 1.0)
    correction = combined_correction(np.zeros(2), [first, second], gain=1.0, maximum_norm=1.0)
    assert np.linalg.norm(correction) == pytest.approx(1.0)
    assert correction[0] > 0 and correction[1] < 0
