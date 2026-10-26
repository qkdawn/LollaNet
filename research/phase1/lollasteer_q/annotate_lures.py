from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from .annotation import ANNOTATOR_FIELDS
from .common import atomic_write_jsonl, read_jsonl


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _packet_labels(packet: dict[str, Any]) -> list[str]:
    options = packet.get("display_options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"packet {packet.get('annotation_id')}: display_options must be non-empty")
    labels = [str(option.get("display_label", "")).strip().upper() for option in options]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"packet {packet.get('annotation_id')}: display labels must be unique")
    return labels


def parse_ranking(value: str, expected_labels: list[str]) -> list[str]:
    labels = [token.upper() for token in re.findall(r"[A-Za-z]+", value)]
    if len(labels) != len(expected_labels) or set(labels) != set(expected_labels):
        expected = " ".join(expected_labels)
        raise ValueError(f"ranking must contain every label exactly once: {expected}")
    return labels


def validate_annotation_progress(
    packets: list[dict[str, Any]], annotations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    packet_by_id = {str(row.get("annotation_id")): row for row in packets}
    if len(packet_by_id) != len(packets) or "None" in packet_by_id:
        raise ValueError("packet contains duplicate or missing annotation_id")
    completed: dict[str, dict[str, Any]] = {}
    for row in annotations:
        if set(row) != ANNOTATOR_FIELDS:
            raise ValueError(f"annotation {row.get('annotation_id')}: fields must equal {sorted(ANNOTATOR_FIELDS)}")
        annotation_id = str(row["annotation_id"])
        if annotation_id in completed:
            raise ValueError(f"duplicate annotation_id: {annotation_id}")
        packet = packet_by_id.get(annotation_id)
        if packet is None or str(row["question_id"]) != str(packet.get("question_id")):
            raise ValueError(f"annotation does not match packet identity: {annotation_id}")
        ranking = row["ranked_display_labels"]
        expected_labels = _packet_labels(packet)
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)) or set(ranking) != set(expected_labels):
            raise ValueError(f"annotation {annotation_id}: ranking is not a complete display-label permutation")
        if not isinstance(row["exclude"], bool):
            raise ValueError(f"annotation {annotation_id}: exclude must be boolean")
        reason = str(row["exclusion_reason"]).strip()
        if row["exclude"] and not reason:
            raise ValueError(f"annotation {annotation_id}: excluded item requires exclusion_reason")
        if not row["exclude"] and reason:
            raise ValueError(f"annotation {annotation_id}: non-excluded item must have an empty exclusion_reason")
        completed[annotation_id] = row
    return completed


def _render_packet(packet: dict[str, Any], position: int, total: int, output_fn: OutputFn) -> None:
    output_fn(f"\n[{position}/{total}] question_id={packet['question_id']}")
    output_fn(str(packet["question"]))
    for option in packet["display_options"]:
        output_fn(f"  {option['display_label']}. {option['text']}")


def run_annotation_session(
    packet_path: Path,
    output_path: Path,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    max_items: int | None = None,
) -> dict[str, Any]:
    packets = read_jsonl(packet_path)
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = validate_annotation_progress(packets, existing)
    packet_order = [str(row["annotation_id"]) for row in packets]
    newly_completed = 0
    quit_requested = False
    for position, packet in enumerate(packets, 1):
        annotation_id = str(packet["annotation_id"])
        if annotation_id in completed:
            continue
        if max_items is not None and newly_completed >= max_items:
            break
        _render_packet(packet, position, len(packets), output_fn)
        expected_labels = _packet_labels(packet)
        while True:
            raw = input_fn(
                f"Rank all labels ({' '.join(expected_labels)}), enter 'exclude <reason>', or 'quit': "
            ).strip()
            if raw.lower() in {"q", "quit"}:
                quit_requested = True
                break
            if raw.lower().startswith("exclude"):
                reason = raw[len("exclude"):].strip()
                if not reason:
                    output_fn("Exclusion requires a reason.")
                    continue
                row = {
                    "annotation_id": annotation_id,
                    "question_id": str(packet["question_id"]),
                    "ranked_display_labels": expected_labels,
                    "exclude": True,
                    "exclusion_reason": reason,
                }
                break
            try:
                ranking = parse_ranking(raw, expected_labels)
            except ValueError as exc:
                output_fn(str(exc))
                continue
            row = {
                "annotation_id": annotation_id,
                "question_id": str(packet["question_id"]),
                "ranked_display_labels": ranking,
                "exclude": False,
                "exclusion_reason": "",
            }
            break
        if quit_requested:
            break
        completed[annotation_id] = row
        newly_completed += 1
        ordered_rows = [completed[key] for key in packet_order if key in completed]
        validate_annotation_progress(packets, ordered_rows)
        atomic_write_jsonl(output_path, ordered_rows)
    remaining = len(packets) - len(completed)
    report = {
        "status": "complete" if remaining == 0 else "needs_author_input",
        "packet": str(packet_path.resolve()),
        "output": str(output_path.resolve()),
        "completed": len(completed),
        "remaining": remaining,
        "newly_completed": newly_completed,
    }
    output_fn(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively complete one blinded LollaSteer-Q lure-ranking packet.")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, help="Optional session cap; output remains resumable.")
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    run_annotation_session(args.packet, args.output, max_items=args.max_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
