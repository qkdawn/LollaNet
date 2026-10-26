"""Prepare route-blind external LLM teacher packets; no labels are generated here."""
from __future__ import annotations

import argparse
from pathlib import Path

from .question_set_v1 import ValidationError, load_semantic_assets, read_jsonl, teacher_packet, write_jsonl


def _validate_sealed_options(include_sealed: bool, confirm_sealed: bool, sealed_output: Path | None) -> None:
    if include_sealed != confirm_sealed:
        raise ValidationError("sealed teacher packets require both --include-sealed and --confirm-sealed")
    if sealed_output is not None and not include_sealed:
        raise ValidationError("--sealed-output is only valid with both sealed confirmation flags")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Open train/development packet JSONL.")
    parser.add_argument("--mode", required=True, choices=("initial", "review"))
    parser.add_argument("--initial-labels", type=Path, help="Required in review mode; one initial label per selected episode.")
    parser.add_argument("--splits", nargs="+", choices=("train", "development", "sealed_test"), help="Optional episode-split filter; defaults to all rows in --episodes.")
    parser.add_argument("--include-sealed", action="store_true", help="First explicit confirmation to materialize selected sealed-test packets.")
    parser.add_argument("--confirm-sealed", action="store_true", help="Second explicit confirmation required with --include-sealed.")
    parser.add_argument("--sealed-output", type=Path, help="Isolated sealed packet JSONL; defaults to <output-parent>/sealed/<output-name> after both confirmations.")
    args = parser.parse_args()

    _validate_sealed_options(args.include_sealed, args.confirm_sealed, args.sealed_output)
    episodes = read_jsonl(args.episodes)
    if args.splits:
        selected = set(args.splits)
        episodes = [episode for episode in episodes if episode.get("split") in selected]

    sealed_episodes = [episode for episode in episodes if episode.get("split") == "sealed_test"]
    if sealed_episodes and not (args.include_sealed and args.confirm_sealed):
        raise ValidationError("selected Episodes include sealed_test; fail closed unless both sealed confirmation flags are present")
    open_episodes = [episode for episode in episodes if episode.get("split") != "sealed_test"]

    sealed_output = args.sealed_output or (args.output.parent / "sealed" / args.output.name)
    if sealed_episodes and sealed_output.resolve() == args.output.resolve():
        raise ValidationError("sealed teacher packets must use a file distinct from the open packet output")
    if sealed_episodes and sealed_output.parent.resolve() == args.output.parent.resolve():
        raise ValidationError("sealed teacher packets must use an isolated directory, not the open packet directory")

    initials = {}
    if args.mode == "review":
        if not args.initial_labels:
            raise ValidationError("--initial-labels is required for review packets")
        for label in read_jsonl(args.initial_labels):
            if label.get("label_pass") == "initial":
                if label.get("episode_id") in initials:
                    raise ValidationError("duplicate initial label for an episode")
                initials[label["episode_id"]] = label

    assets = load_semantic_assets()

    def packets_for(rows: list[dict]) -> list[dict]:
        packets = []
        for episode in rows:
            initial = initials.get(episode["episode_id"])
            if args.mode == "review" and initial is None:
                raise ValidationError("review packet missing matching initial label")
            packets.append(teacher_packet(episode, args.mode, initial, assets=assets))
        return packets

    write_jsonl(args.output, packets_for(open_episodes))
    if sealed_episodes:
        write_jsonl(sealed_output, packets_for(sealed_episodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
