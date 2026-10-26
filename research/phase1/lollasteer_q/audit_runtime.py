from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_audit import audit_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify LollaSteer-Q runtime artifacts under the current protocol.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_runtime(args.runtime_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
