from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import validate_annotation_packets


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate blinded lure-ranking packets and withheld mappings.")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_annotation_packets(args.annotation_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
