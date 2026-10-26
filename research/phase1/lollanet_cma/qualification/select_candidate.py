from __future__ import annotations

import argparse
import json
from pathlib import Path

from .selection import select_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the frozen LollaNet-CMA base model from complete qualification runs.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("qualification_charter.yaml"))
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_candidate(args.charter, args.run_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

