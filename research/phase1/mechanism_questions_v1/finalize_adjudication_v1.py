"""CLI for preparing and finalizing the adjudication child chain."""
from __future__ import annotations
import argparse
from pathlib import Path
from .adjudication_chain_v1 import build_adjudication_packets, finalize_adjudication

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dataset-root", required=True, type=Path); p.add_argument("--parent-manifest", required=True, type=Path); p.add_argument("--output-dir", required=True, type=Path); p.add_argument("--review-visibility", choices=("route_blind", "status_visible", "status_and_rationale_visible"), default="route_blind")
    f = sub.add_parser("finalize"); f.add_argument("--dataset-root", required=True, type=Path); f.add_argument("--parent-manifest", required=True, type=Path); f.add_argument("--packets", required=True, type=Path); f.add_argument("--labels", required=True, type=Path); f.add_argument("--audit-ledger", required=True, type=Path); f.add_argument("--policy", required=True, type=Path); f.add_argument("--prompt-contract", required=True, type=Path); f.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_adjudication_packets(parent_manifest_path=args.parent_manifest, dataset_root=args.dataset_root, output_dir=args.output_dir, review_visibility=args.review_visibility)
        print(f"Prepared {len(result['eligible_episode_ids'])} packets: {result['packet_path']}"); return 0
    result = finalize_adjudication(parent_manifest_path=args.parent_manifest, dataset_root=args.dataset_root, packets_path=args.packets, labels_path=args.labels, audit_ledger_path=args.audit_ledger, policy_path=args.policy, prompt_contract_path=args.prompt_contract, output_dir=args.output_dir)
    print(f"Finalized {len(result['accepted_episode_ids'])} targets: {args.output_dir}"); return 0

if __name__ == "__main__": raise SystemExit(main())
