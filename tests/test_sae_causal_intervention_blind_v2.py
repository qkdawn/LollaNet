from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal.intervention_blind_v2 import _alias


def test_blind_alias_does_not_leak_condition() -> None:
    alias = _alias("smoke__case__axis__feature-1__dose-+2")
    assert alias.startswith("causal-")
    assert "feature" not in alias
    assert "dose" not in alias
