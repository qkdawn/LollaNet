from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import matched_dynamics


def audit_result(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = result["records"]
    dynamics = matched_dynamics(records)
    anchor_coverage = [float(row["anchor_coverage"]) for row in records]
    output_tokens = [int(row["output_tokens"]) for row in records]
    maximum = int(result.get("maximum_new_tokens", max(output_tokens)))
    return {
        "type": "target_attractor_smoke_audit_v1",
        "source": str(result_path),
        "formal_evaluation_records_accessed": int(result["formal_evaluation_records_accessed"]),
        "matched_dynamics": dynamics,
        "zero_anchor_coverage_rate": sum(value == 0 for value in anchor_coverage) / len(anchor_coverage),
        "maximum_token_hit_rate": sum(value == maximum for value in output_tokens) / len(output_tokens),
        "candidate_internal_recovery": bool(
            dynamics["mean_perturbation_effect_vs_unperturbed"] is not None
            and float(dynamics["mean_perturbation_effect_vs_unperturbed"]) < 0
            and dynamics["mean_recovery_fraction"] is not None
            and float(dynamics["mean_recovery_fraction"]) > 0.5
        ),
        "target_attractor_supported": False,
        "reason": "behavioral selectivity and task validity gate failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_result(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
