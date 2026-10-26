from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_development_selection


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen LollaNet-CMA development selection.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("development_charter.yaml"))
    parser.add_argument("--access-view", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_development_selection(
        args.charter, args.access_view, args.direction_manifest, args.preflight, args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["hard_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

