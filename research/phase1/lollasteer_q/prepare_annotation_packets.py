from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation import prepare_annotation_packets


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare blinded human lure-ranking packets.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pilot.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--annotator", action="append", dest="annotators")
    args = parser.parse_args()
    annotators = tuple(args.annotators) if args.annotators else ("annotator_a", "annotator_b", "annotator_c")
    print(json.dumps(prepare_annotation_packets(args.config, args.output, annotators, args.source_jsonl), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
