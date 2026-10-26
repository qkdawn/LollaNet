from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import prepare_primary_annotation_delivery


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare isolated packets for the two independent primary annotators.")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_primary_annotation_delivery(args.annotation_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
