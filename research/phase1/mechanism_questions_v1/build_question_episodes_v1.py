"""Build open Episodes and optionally isolate sealed-test Episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_question_conditioning_v1 import audit_question_conditioning
from .question_set_v1 import ValidationError, build_episodes, read_jsonl, write_jsonl


def _validate_sealed_options(include_sealed: bool, confirm_sealed: bool, sealed_output: Path | None) -> None:
    if include_sealed != confirm_sealed:
        raise ValidationError("sealed Episodes require both --include-sealed and --confirm-sealed")
    if sealed_output is not None and not include_sealed:
        raise ValidationError("--sealed-output is only valid with both sealed confirmation flags")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-packs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Open train/development output directory.")
    parser.add_argument("--require-v1-size", action="store_true", help="Validate the complete 160-pack/640-Episode source corpus before partitioning output.")
    parser.add_argument("--include-sealed", action="store_true", help="First explicit confirmation to materialize sealed-test Episodes.")
    parser.add_argument("--confirm-sealed", action="store_true", help="Second explicit confirmation required with --include-sealed.")
    parser.add_argument("--sealed-output", type=Path, help="Isolated sealed output directory; defaults to <output>/sealed after both confirmations.")
    args = parser.parse_args()

    _validate_sealed_options(args.include_sealed, args.confirm_sealed, args.sealed_output)
    sealed_root = args.sealed_output or (args.output / "sealed")
    if args.include_sealed and sealed_root.resolve() == args.output.resolve():
        raise ValidationError("sealed Episodes must use a directory distinct from the open output directory")
    if args.include_sealed and _is_within(sealed_root, args.output) and not _is_within(sealed_root, args.output / "sealed"):
        raise ValidationError("sealed Episodes inside the open output must stay under <output>/sealed")
    default_sealed_root = args.output / "sealed"
    if default_sealed_root.exists() and (not args.include_sealed or sealed_root.resolve() != default_sealed_root.resolve()):
        raise ValidationError("open output already contains a different sealed subtree; use a clean output directory")

    episodes = build_episodes(read_jsonl(args.fact_packs), require_v1_size=args.require_v1_size)
    open_episodes = [row for row in episodes if row["split"] != "sealed_test"]
    sealed_episodes = [row for row in episodes if row["split"] == "sealed_test"]

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "episodes.jsonl", open_episodes)
    conditioning_audit = audit_question_conditioning(open_episodes)
    conditioning_audit_path = args.output / "question_conditioning_audit.json"
    conditioning_audit_path.write_text(
        json.dumps(conditioning_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifact": "lollanet-mechanism-questions-v1",
        "fact_pack_input": str(args.fact_packs),
        "episodes": len(open_episodes),
        "case_packs": len({row["case_id"] for row in open_episodes}),
        "splits": {split: sum(row["split"] == split for row in open_episodes) for split in ("train", "development")},
        "full_v1_source_validated": bool(args.require_v1_size),
        "question_conditioning_audit": str(conditioning_audit_path),
        "question_conditioning_checks": conditioning_audit["checks"],
        "sealed_included": bool(args.include_sealed),
        "sealed_artifact_root": str(sealed_root) if args.include_sealed else None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.include_sealed:
        sealed_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(sealed_root / "episodes.jsonl", sealed_episodes)
        sealed_manifest = {
            "artifact": "lollanet-mechanism-questions-v1-sealed",
            "fact_pack_input": str(args.fact_packs),
            "split": "sealed_test",
            "episodes": len(sealed_episodes),
            "case_packs": len({row["case_id"] for row in sealed_episodes}),
            "double_confirmation": True,
        }
        (sealed_root / "manifest.json").write_text(json.dumps(sealed_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
