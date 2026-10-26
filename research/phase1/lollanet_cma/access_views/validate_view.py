from __future__ import annotations

import argparse
import json
from pathlib import Path

from .views import validate_access_view


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a split-scoped LollaNet-CMA access view.")
    parser.add_argument("view", type=Path)
    parser.add_argument("--expected-split")
    args = parser.parse_args()
    print(json.dumps(validate_access_view(args.view, args.expected_split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

