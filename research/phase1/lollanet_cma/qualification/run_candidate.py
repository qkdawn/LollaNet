from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one preregistered LollaNet-CMA base-model qualification candidate.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("qualification_charter.yaml"))
    parser.add_argument("--access-view", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate(args.charter, args.access_view, args.readiness, args.candidate, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
