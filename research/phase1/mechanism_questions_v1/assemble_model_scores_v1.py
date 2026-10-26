"""Assemble two route-blind raw score files into manifest-ready labels and ledger."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .model_score_chain_v1 import make_model_score_label, model_score_ledger_entry
from .question_set_v1 import read_jsonl, write_jsonl
from .teacher_chain_v1 import make_teacher_identity


def _raw_by_episode(path: Path, expected_ids: list[str], model_ids: list[str], slot: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != len(expected_ids):
        raise ValueError(f"teacher {slot} raw score row count mismatch")
    result: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if set(row) != {"episode_id", "scores", "confidence", "rationale"}:
            raise ValueError(f"teacher {slot} raw row must contain exactly episode_id/scores/confidence/rationale")
        episode_id = row["episode_id"]
        if episode_id != expected_ids[position]:
            raise ValueError(f"teacher {slot} raw score order/id mismatch at row {position + 1}")
        if episode_id in result:
            raise ValueError(f"duplicate teacher {slot} score for {episode_id}")
        scores = row["scores"]
        if not isinstance(scores, dict) or set(scores) != set(model_ids):
            raise ValueError(f"teacher {slot} must score exactly all models for {episode_id}")
        result[episode_id] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--raw-a", required=True, type=Path)
    parser.add_argument("--raw-b", required=True, type=Path)
    parser.add_argument("--output-labels", required=True, type=Path)
    parser.add_argument("--output-ledger", required=True, type=Path)
    for slot in ("a", "b"):
        parser.add_argument(f"--provider-{slot}", required=True)
        parser.add_argument(f"--model-{slot}", required=True)
        parser.add_argument(f"--deployment-{slot}", required=True)
        parser.add_argument(f"--version-{slot}", required=True)
        parser.add_argument(f"--run-id-{slot}", required=True)
    args = parser.parse_args()
    if args.output_labels.exists() or args.output_ledger.exists():
        raise FileExistsError("refusing to overwrite assembled score labels or ledger")
    packets = read_jsonl(args.packets)
    if not packets:
        raise ValueError("score packet file is empty")
    expected_ids = [packet["episode_id"] for packet in packets]
    model_ids = packets[0]["model_ids"]
    if any(packet.get("model_ids") != model_ids for packet in packets):
        raise ValueError("score packet model order is not stable")
    raw = {
        "a": _raw_by_episode(args.raw_a, expected_ids, model_ids, "a"),
        "b": _raw_by_episode(args.raw_b, expected_ids, model_ids, "b"),
    }
    identities = {
        slot: make_teacher_identity(
            provider=getattr(args, f"provider_{slot}"),
            model=getattr(args, f"model_{slot}"),
            deployment=getattr(args, f"deployment_{slot}"),
            version=getattr(args, f"version_{slot}"),
        )
        for slot in ("a", "b")
    }
    labels = []
    for packet in packets:
        for slot in ("a", "b"):
            row = raw[slot][packet["episode_id"]]
            labels.append(make_model_score_label(
                packet=packet,
                slot=slot,
                teacher_identity=identities[slot],
                scores=row["scores"],
                confidence=row["confidence"],
                rationale=row["rationale"],
                raw_response=row,
                api_response_id=f"codex-agent::{getattr(args, f'run_id_{slot}')}::{packet['episode_id']}",
                run_id=getattr(args, f"run_id_{slot}"),
            ))
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_labels, labels)
    write_jsonl(args.output_ledger, [model_score_ledger_entry(label) for label in labels])
    print(f"Assembled {len(labels)} labels and {len(labels)} ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
