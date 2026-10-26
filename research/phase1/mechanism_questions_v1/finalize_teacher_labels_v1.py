"""Validate the full two-pass teacher chain and materialize trusted Router targets."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_teacher_v1 import INITIAL_PROMPT_VERSION, REVIEW_PROMPT_VERSION
from .question_set_v1 import read_jsonl, write_jsonl
from .teacher_chain_v1 import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    file_descriptor,
    validate_teacher_chain,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finalize_teacher_labels(
    *,
    episodes_path: Path,
    initial_labels_path: Path,
    review_labels_path: Path,
    initial_audit_ledger_path: Path,
    review_audit_ledger_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate labels plus both ledgers, then write accepted labels and a hash manifest."""
    episodes_path = Path(episodes_path)
    initial_labels_path = Path(initial_labels_path)
    review_labels_path = Path(review_labels_path)
    initial_audit_ledger_path = Path(initial_audit_ledger_path)
    review_audit_ledger_path = Path(review_audit_ledger_path)
    output_dir = Path(output_dir)

    episodes = read_jsonl(episodes_path)
    initial = read_jsonl(initial_labels_path)
    review = read_jsonl(review_labels_path)
    initial_ledger = read_jsonl(initial_audit_ledger_path)
    review_ledger = read_jsonl(review_audit_ledger_path)
    summary = validate_teacher_chain(
        episodes,
        initial,
        review,
        initial_ledger,
        review_ledger,
        expected_initial_prompt=INITIAL_PROMPT_VERSION,
        expected_review_prompt=REVIEW_PROMPT_VERSION,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_labels_path = output_dir / "all_labels.jsonl"
    accepted_path = output_dir / "accepted_review.jsonl"
    write_jsonl(all_labels_path, initial + review)
    write_jsonl(accepted_path, summary["accepted"])

    paths_and_rows = {
        "episodes": (episodes_path, len(episodes)),
        "initial_labels": (initial_labels_path, len(initial)),
        "review_labels": (review_labels_path, len(review)),
        "initial_audit_ledger": (initial_audit_ledger_path, len(initial_ledger)),
        "review_audit_ledger": (review_audit_ledger_path, len(review_ledger)),
        "all_labels": (all_labels_path, len(initial) + len(review)),
        "accepted_reviews": (accepted_path, len(summary["accepted"])),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "files": {
            role: file_descriptor(path, relative_to=output_dir, rows=rows)
            for role, (path, rows) in paths_and_rows.items()
        },
        "episode_ids": summary["episode_ids"],
        "accepted_episode_ids": summary["accepted_episode_ids"],
        "prompt_versions": summary["prompt_versions"],
        "teacher_identities": summary["teacher_identities"],
    }
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--initial-labels", required=True, type=Path)
    parser.add_argument("--review-labels", required=True, type=Path)
    parser.add_argument("--initial-audit-ledger", required=True, type=Path)
    parser.add_argument("--review-audit-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    finalize_teacher_labels(
        episodes_path=args.episodes,
        initial_labels_path=args.initial_labels,
        review_labels_path=args.review_labels,
        initial_audit_ledger_path=args.initial_audit_ledger,
        review_audit_ledger_path=args.review_audit_ledger,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
