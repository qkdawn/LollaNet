from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import migrate_unstarted_annotation_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate an unstarted lure annotation bundle to the minimal v1 contract.")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = migrate_unstarted_annotation_contract(args.annotation_dir, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
