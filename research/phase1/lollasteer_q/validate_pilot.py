from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate import validate_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LollaSteer-Q pilot bundle and leakage boundaries.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = validate_bundle(args.bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
