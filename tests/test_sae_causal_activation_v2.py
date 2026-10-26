from __future__ import annotations

from research.phase1.target_attractor_pilot.sae_causal import extract_v2


def test_v2_extractor_uses_frozen_chat_renderer() -> None:
    assert extract_v2.render_prompt.__module__.endswith("template_attractor_pilot.pilot")
