from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import validate_primary_annotation_delivery


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate isolated primary annotation delivery packets.")
    parser.add_argument("--delivery-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_primary_annotation_delivery(args.delivery_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
