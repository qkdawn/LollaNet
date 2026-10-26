from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.subspace_interchange_smoke_v4 import (
    norm_match,
    subspace_swap_update,
)


def test_subspace_swap_and_random_norm_matching() -> None:
    basis = np.asarray([[1.0], [0.0]])
    update = subspace_swap_update(
        np.asarray([[3.0, 4.0]]), np.asarray([[1.0, 1.0]]), basis
    )
    np.testing.assert_allclose(update, [[2.0, 0.0]])
    matched, scale = norm_match(np.asarray([[0.0, 4.0]]), update)
    np.testing.assert_allclose(np.linalg.norm(matched), np.linalg.norm(update))
    assert scale == 0.5
