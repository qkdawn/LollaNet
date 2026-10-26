from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import prepare_tiebreak_packet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a blinded third-annotator packet containing only primary top-choice disagreements."
    )
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_tiebreak_packet(args.annotation_dir, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
