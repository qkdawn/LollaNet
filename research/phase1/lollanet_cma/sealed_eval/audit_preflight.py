from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import audit_sealed_evaluation_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sealed LollaNet-CMA evaluation readiness.")
    parser.add_argument(
        "--charter", type=Path, default=Path(__file__).with_name("sealed_eval_charter.yaml"),
    )
    parser.add_argument(
        "--control-charter", type=Path, default=Path(__file__).with_name("control_set_charter.yaml"),
    )
    parser.add_argument("--evaluation-release", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_sealed_evaluation_preflight(
        args.charter, args.control_charter, args.evaluation_release, args.probe_manifest, args.output,
    )
    print(result["sealed_evaluation_gate_status"])


if __name__ == "__main__":
    main()
