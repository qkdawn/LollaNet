from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import generate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate behavior, interaction, and pilot go/no-go reports.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pilot.yaml"))
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = generate_report(args.config, args.evaluator, args.run_output, args.output, args.selection)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
