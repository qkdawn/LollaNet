"""Build V1.1 open artifacts and optionally isolate sealed-test artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .audit_question_conditioning_v1 import audit_question_conditioning
from .public_corpus_v1 import (
    DEFAULT_RUNTIME_ROOT,
    build_all_fact_packs,
    source_cache_manifest,
    write_public_cases,
    write_source_ledger,
)
from .question_set_v1 import ValidationError, build_episodes, load_semantic_assets, teacher_packet, write_jsonl


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _checksums(root: Path, excluded_roots: Iterable[Path] = ()) -> dict[str, dict[str, object]]:
    excluded = tuple(path.resolve() for path in excluded_roots)
    result = {}
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if any(_is_within(resolved, blocked) for blocked in excluded):
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_file() and path.name != "checksums.json" and "source-cache" not in path.parts:
            data = path.read_bytes()
            result[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return result


def _validate_sealed_options(include_sealed: bool, confirm_sealed: bool, sealed_output: Path | None) -> None:
    if include_sealed != confirm_sealed:
        raise ValidationError("sealed artifacts require both --include-sealed and --confirm-sealed")
    if sealed_output is not None and not include_sealed:
        raise ValidationError("--sealed-output is only valid with both sealed confirmation flags")


def _guard_open_root(root: Path, include_sealed: bool, sealed_root: Path) -> None:
    legacy_paths = (root / "episodes" / "sealed_test.jsonl", root / "teacher_packets" / "sealed_test")
    if any(path.exists() for path in legacy_paths):
        raise ValidationError("open output root contains legacy sealed artifacts; move them to an isolated sealed root first")
    default_sealed_root = root / "sealed"
    if default_sealed_root.exists() and (not include_sealed or sealed_root.resolve() != default_sealed_root.resolve()):
        raise ValidationError("open output root already contains a different sealed subtree; use a clean open root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--skip-public-cases", action="store_true")
    parser.add_argument(
        "--source-runtime-root", type=Path,
        help="External runtime root containing source-cache; defaults to a sibling of --output and must not be inside the training dataset root.",
    )
    parser.add_argument("--include-sealed", action="store_true", help="First explicit confirmation to materialize sealed-test artifacts.")
    parser.add_argument("--confirm-sealed", action="store_true", help="Second explicit confirmation required with --include-sealed.")
    parser.add_argument("--sealed-output", type=Path, help="Isolated sealed artifact root; defaults to <output>/sealed after both confirmations.")
    args = parser.parse_args()

    _validate_sealed_options(args.include_sealed, args.confirm_sealed, args.sealed_output)
    root = args.output
    source_runtime_root = args.source_runtime_root or (root.parent / f".{root.name}-source-runtime")
    sealed_root = args.sealed_output or (root / "sealed")
    _guard_open_root(root, args.include_sealed, sealed_root)
    root.mkdir(parents=True, exist_ok=True)
    if source_runtime_root.resolve() == root.resolve() or _is_within(source_runtime_root, root):
        raise ValidationError("source runtime/cache must be outside the open training dataset root")
    source_runtime_root.mkdir(parents=True, exist_ok=True)

    if args.include_sealed and sealed_root.resolve() == root.resolve():
        raise ValidationError("sealed artifacts must use a root distinct from the open output root")
    if args.include_sealed and _is_within(sealed_root, root) and not _is_within(sealed_root, root / "sealed"):
        raise ValidationError("sealed artifacts inside the open root must stay under <output>/sealed")

    packs = build_all_fact_packs(source_runtime_root)
    episodes = build_episodes(packs, require_v1_size=True)
    split_by_case = {row["case_id"]: row["split"] for row in episodes}
    open_packs = [pack for pack in packs if split_by_case[pack["case_id"]] != "sealed_test"]
    sealed_packs = [pack for pack in packs if split_by_case[pack["case_id"]] == "sealed_test"]
    open_episodes = [row for row in episodes if row["split"] != "sealed_test"]
    sealed_episodes = [row for row in episodes if row["split"] == "sealed_test"]

    # The open root never receives sealed fact packs, ledger rows, Episodes, or teacher packets.
    write_jsonl(root / "fact_packs.jsonl", open_packs)
    write_source_ledger(root, open_packs)
    ledger_path = root / "source_ledger.jsonl"
    case_paths = [] if args.skip_public_cases else write_public_cases(open_packs)
    write_jsonl(root / "build" / "episodes.jsonl", open_episodes)
    for split in ("train", "development"):
        write_jsonl(root / "episodes" / f"{split}.jsonl", [row for row in open_episodes if row["split"] == split])
    write_jsonl(root / "episodes" / "train_development.jsonl", open_episodes)
    conditioning_audit = audit_question_conditioning(open_episodes)
    conditioning_audit_path = root / "reports" / "question_conditioning_audit.json"
    _write_json(conditioning_audit_path, conditioning_audit)

    assets = load_semantic_assets()
    write_jsonl(
        root / "teacher_packets" / "train_development" / "initial.jsonl",
        [teacher_packet(row, "initial", assets=assets) for row in open_episodes],
    )

    if args.include_sealed:
        sealed_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(sealed_root / "fact_packs.jsonl", sealed_packs)
        write_source_ledger(sealed_root, sealed_packs)
        write_jsonl(sealed_root / "episodes" / "sealed_test.jsonl", sealed_episodes)
        write_jsonl(
            sealed_root / "teacher_packets" / "initial.jsonl",
            [teacher_packet(row, "initial", assets=assets) for row in sealed_episodes],
        )
        sealed_manifest = {
            "artifact": "lollanet-mechanism-question-set-v1.1-sealed",
            "split": "sealed_test",
            "fact_packs": len(sealed_packs),
            "episodes_materialized": len(sealed_episodes),
            "source_ledger": str(sealed_root / "source_ledger.jsonl"),
            "double_confirmation": True,
            "synthetic_or_hash_teacher_labels": False,
        }
        _write_json(sealed_root / "manifest.json", sealed_manifest)
        _write_json(sealed_root / "checksums.json", _checksums(sealed_root))

    manifest = {
        "artifact": "lollanet-mechanism-question-set-v1.1",
        "episode_state_contract": "fact- and question-conditioned task_relevance; declared counterfactual state deltas applied",
        "question_conditioning_audit": str(conditioning_audit_path),
        "question_conditioning_checks": conditioning_audit["checks"],
        "teacher_label_status": "pending_external_independent_llm_initial_and_audit",
        "selected_splits": ["development", "train"],
        "episodes_materialized": len(open_episodes),
        "fact_packs": len(open_packs),
        "full_corpus_fact_packs_built": len(packs),
        "public_case_paths": [str(path) for path in case_paths],
        "source_ledger": str(ledger_path),
        "source_runtime_root": str(source_runtime_root),
        "source_cache": source_cache_manifest(source_runtime_root),
        "sealed_included": bool(args.include_sealed),
        "sealed_artifact_root": str(sealed_root) if args.include_sealed else None,
        "synthetic_or_hash_teacher_labels": False,
    }
    _write_json(root / "manifest.json", manifest)
    excluded = [sealed_root] if args.include_sealed and _is_within(sealed_root, root) else []
    _write_json(root / "checksums.json", _checksums(root, excluded_roots=excluded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
