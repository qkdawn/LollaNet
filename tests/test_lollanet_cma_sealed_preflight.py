from __future__ import annotations

from pathlib import Path

from research.phase1.lollanet_cma.common import stable_hash
from research.phase1.lollanet_cma.sealed_eval.preflight import audit_sealed_evaluation_preflight


def test_sealed_evaluation_stays_blocked_without_release(tmp_path: Path):
    charter = Path("research/phase1/lollanet_cma/sealed_eval/sealed_eval_charter.yaml")
    control_charter = Path("research/phase1/lollanet_cma/sealed_eval/control_set_charter.yaml")
    output = tmp_path / "preflight.json"
    result = audit_sealed_evaluation_preflight(
        charter, control_charter, tmp_path / "missing-release.json",
        tmp_path / "missing-probe.json", output,
    )
    assert result["sealed_evaluation_gate_status"] == "blocked_missing_release"
    assert result["evaluation_authorized"] is False
    assert result["evaluation_release"] is None
    assert result["view_validation"] is None
    content = dict(result)
    claimed = content.pop("sealed_evaluation_preflight_content_sha256")
    assert claimed == stable_hash(content)
