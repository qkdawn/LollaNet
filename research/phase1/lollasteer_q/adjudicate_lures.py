from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import adjudicate_lures


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate lure rankings without inspecting target-model outputs.")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-count-per-category", type=int, default=15)
    parser.add_argument("--third-annotations", type=Path)
    parser.add_argument("--frozen-at", required=True, help="Author-supplied ISO-8601 adjudication timestamp.")
    parser.add_argument("--report", type=Path, help="Optional adjudication report path.")
    args = parser.parse_args()
    report = adjudicate_lures(
        args.annotation_dir,
        args.mapping,
        args.output,
        args.final_count_per_category,
        args.third_annotations,
        args.frozen_at,
        args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
