from __future__ import annotations

import argparse
import json
from pathlib import Path

from .freeze import freeze_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify one frozen LollaNet-CMA qualification candidate.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("qualification_charter.yaml"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    result = freeze_candidate(args.charter, args.candidate, args.endpoint, args.max_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
