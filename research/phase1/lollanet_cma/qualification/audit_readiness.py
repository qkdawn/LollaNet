from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import atomic_write_json
from .readiness import audit_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LollaNet-CMA base-model qualification readiness.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("qualification_charter.yaml"))
    parser.add_argument("--stage0-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_readiness(args.charter, args.stage0_bundle)
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

