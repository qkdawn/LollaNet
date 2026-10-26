from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate import validate_stage0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen LollaNet-CMA Stage 0 data and leakage contracts.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = validate_stage0(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

