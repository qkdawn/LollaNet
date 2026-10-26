"""CLI for dense dual-teacher thought-model score supervision."""
from __future__ import annotations

import argparse
from pathlib import Path

from .model_score_chain_v1 import build_model_score_packets, finalize_model_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset-root", required=True, type=Path)
    prepare.add_argument("--source-manifest", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--max-score-gap", default=1.0, type=float)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--dataset-root", required=True, type=Path)
    finalize.add_argument("--source-manifest", required=True, type=Path)
    finalize.add_argument("--packets", required=True, type=Path)
    finalize.add_argument("--labels", required=True, type=Path)
    finalize.add_argument("--audit-ledger", required=True, type=Path)
    finalize.add_argument("--policy", required=True, type=Path)
    finalize.add_argument("--prompt-contract", required=True, type=Path)
    finalize.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_model_score_packets(
            source_manifest_path=args.source_manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            max_score_gap=args.max_score_gap,
        )
        print(f"Prepared {len(result['episode_ids'])} dense score packets: {result['packet_path']}")
        return 0
    result = finalize_model_scores(
        source_manifest_path=args.source_manifest,
        dataset_root=args.dataset_root,
        packets_path=args.packets,
        labels_path=args.labels,
        audit_ledger_path=args.audit_ledger,
        policy_path=args.policy,
        prompt_contract_path=args.prompt_contract,
        output_dir=args.output_dir,
    )
    print(f"Finalized dense score targets for {len(result['aggregated_episode_ids'])} Episodes: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
