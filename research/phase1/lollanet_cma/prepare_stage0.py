from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage0 import prepare_stage0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the frozen grouped CLADDER Stage 0 bundle for LollaNet-CMA.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("experiment_charter.yaml"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-split-manifest", type=Path)
    args = parser.parse_args()
    result = prepare_stage0(args.charter, args.source_root, args.output, args.frozen_split_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
