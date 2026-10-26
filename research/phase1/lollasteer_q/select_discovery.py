from __future__ import annotations

import argparse
import json
from pathlib import Path

from .selection import select_discovery_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze CAA and low-rank hyperparameters using discovery-only runs.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pilot.yaml"))
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_discovery_runs(args.config, args.evaluator, args.baseline_output, args.candidate_output, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
