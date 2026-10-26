from __future__ import annotations

import argparse
import json
from pathlib import Path

from .views import create_access_view


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a split-scoped LollaNet-CMA access view.")
    parser.add_argument("--stage0-bundle", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_access_view(args.stage0_bundle, args.split, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

