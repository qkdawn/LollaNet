from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic LollaSteer-Q pilot bundle.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pilot.yaml"))
    parser.add_argument("--lures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_pilot(args.config, args.lures, args.output, args.source_jsonl), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
