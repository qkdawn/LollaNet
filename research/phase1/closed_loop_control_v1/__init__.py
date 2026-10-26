"""Closed-loop latent control primitives for open-ended generation."""

from .control import (
    AXES,
    ClosedLoopConfig,
    ClosedLoopGenerationHook,
    EnsembleLinearObserver,
    LowRankFeedbackPolicy,
    ObserverEstimate,
    Perturbation,
    fit_group_bootstrap_observer,
    recovery_metrics,
)

__all__ = [
    "AXES",
    "ClosedLoopConfig",
    "ClosedLoopGenerationHook",
    "EnsembleLinearObserver",
    "LowRankFeedbackPolicy",
    "ObserverEstimate",
    "Perturbation",
    "fit_group_bootstrap_observer",
    "recovery_metrics",
]

