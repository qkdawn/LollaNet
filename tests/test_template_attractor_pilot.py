from __future__ import annotations

import numpy as np

from research.phase1.template_attractor_pilot.data import build_cases, paired_rows, render_natural, render_template
from research.phase1.template_attractor_pilot.pilot import choose_threshold, generated_metrics, response_positions


def test_case_splits_hold_out_template_families() -> None:
    cases = build_cases()
    by_split = {split: {case.template_family for case in cases if case.split == split} for split in ("train", "development", "evaluation")}
    assert len(cases) == 24
    assert by_split["train"].isdisjoint(by_split["development"])
    assert by_split["train"].isdisjoint(by_split["evaluation"])
    assert by_split["development"].isdisjoint(by_split["evaluation"])


def test_pairs_preserve_case_content_and_labels() -> None:
    cases = build_cases()
    rows = paired_rows(cases)
    assert len(rows) == 48
    for case in cases:
        pair = [row for row in rows if row["case_id"] == case.case_id]
        assert {row["label"] for row in pair} == {0, 1}
        for anchor in case.anchors:
            assert anchor.lower() in render_natural(case).lower()
            assert anchor.lower() in render_template(case).lower()


def test_positions_and_threshold_are_bounded() -> None:
    positions = response_positions(10, 6)
    assert positions == {"prompt_end": 9, "full": 15, "early_4": 13, "early_8": 15, "early_16": 15}
    threshold = choose_threshold(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.6, 0.9]))
    assert 0.4 < threshold <= 0.6


def test_generation_metrics_distinguish_surface_template_without_losing_anchors() -> None:
    case = build_cases()[0]
    natural = generated_metrics(render_natural(case), case.anchors)
    template = generated_metrics(render_template(case), case.anchors)
    assert template["template_marker_density"] > natural["template_marker_density"]
    assert template["anchor_coverage"] == natural["anchor_coverage"] == 1.0
