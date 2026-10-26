"""Validate question-set Episodes and two-pass external Weak teacher labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .question_set_v1 import coverage_report, read_jsonl, validate_episodes, validate_quality_gates, validate_teacher_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-v1-size", action="store_true")
    args = parser.parse_args()
    episodes = read_jsonl(args.episodes)
    validate_episodes(episodes, require_v1_size=args.require_v1_size)
    report = {"episodes_valid": len(episodes), "full_v1_validated": bool(args.require_v1_size)}
    if args.labels:
        labels = read_jsonl(args.labels)
        validate_teacher_labels(labels, episodes, require_complete=args.require_v1_size)
        report["labels_valid"] = len(labels)
        report["coverage"] = coverage_report(episodes, labels)
        if args.require_v1_size:
            validate_quality_gates(report["coverage"])
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
