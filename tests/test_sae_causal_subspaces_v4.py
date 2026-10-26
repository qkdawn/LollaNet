from __future__ import annotations

import numpy as np

from research.phase1.target_attractor_pilot.sae_causal.fit_subspaces_v4 import (
    orthonormalize,
    project_delta,
)


def test_projection_uses_the_whole_subspace() -> None:
    basis = orthonormalize(np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    projected = project_delta(np.asarray([2.0, 3.0, 7.0]), basis)
    np.testing.assert_allclose(projected, [2.0, 3.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)
