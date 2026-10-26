"""Build the external Episode-level microconcept labels v1 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.phase1.microconcepts.microconcept_labels_v1 import (
    annotation_guideline,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    build_state_labels,
    build_task_labels,
    canonical_jsonl,
    distribution_report,
    frozen_test_rows,
    load_cards,
    load_claims,
    load_episodes,
    select_pilot_and_freeze,
)

DEFAULT_CANDIDATES = Path(r"D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl")
DEFAULT_CLAIMS = Path(r"D:\LollaNetData\mda-claims-v1\claims.jsonl")
DEFAULT_OUTPUT = Path(r"D:\LollaNetData\microconcept-labels-v1")


def compatible_resume(manifest_path: Path, candidates: Path, claims: Path) -> None:
    """Fail closed when a resumed output folder belongs to different source inputs."""
    if not manifest_path.exists():
        return
    from research.phase1.microconcepts.microconcept_labels_v1 import sha256_file
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_files", {})
    expected_candidates = source.get("candidates", {}).get("sha256")
    expected_claims = source.get("mda_claims", {}).get("sha256")
    if expected_candidates and expected_candidates != sha256_file(candidates):
        raise ValueError("--resume refused: candidate input SHA-256 differs from existing manifest")
    if expected_claims and expected_claims != sha256_file(claims):
        raise ValueError("--resume refused: MD&A claim input SHA-256 differs from existing manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="verify prior source hashes before deterministic repair/rebuild")
    args = parser.parse_args()
    if not args.candidates.exists() or not args.claims.exists():
        raise FileNotFoundError("candidate and claim inputs must exist")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.resume:
        compatible_resume(manifest_path, args.candidates, args.claims)

    cards = load_cards()
    # Freeze the temporal test selection before the label creation calls below.
    episodes = load_episodes(args.candidates)
    allowed_claim_ids = {claim_id for card in cards.values() for claim_id in card.claims}
    claims_by_accession = load_claims(args.claims, allowed_claim_ids)
    pilot, frozen, selection = select_pilot_and_freeze(episodes, claims_by_accession)
    labels, queue = build_state_labels(pilot, cards, claims_by_accession)
    tasks = build_task_labels(labels, cards)
    report = distribution_report(labels, tasks, queue, selection)

    paths = {
        "labels": args.output / "episode_microconcept_labels.jsonl",
        "tasks": args.output / "task_relevance_labels.jsonl",
        "guideline": args.output / "annotation_guideline.md",
        "queue": args.output / "human_review_queue.jsonl",
        "report": args.output / "agreement_and_distribution_report.json",
        "frozen": args.output / "frozen_time_test_episodes.jsonl",
        "validation": args.output / "validation_report.json",
    }
    atomic_write_text(paths["labels"], canonical_jsonl(labels))
    atomic_write_text(paths["tasks"], canonical_jsonl(tasks))
    atomic_write_text(paths["guideline"], annotation_guideline(cards))
    atomic_write_text(paths["queue"], canonical_jsonl(queue))
    atomic_write_json(paths["report"], report)
    atomic_write_text(paths["frozen"], canonical_jsonl(frozen_test_rows(frozen)))

    # Initial manifest permits the validator to verify source provenance. The
    # final manifest adds the validator report's immutable hash after validation.
    manifest_files = [paths["labels"], paths["tasks"], paths["guideline"], paths["queue"], paths["report"], paths["frozen"]]
    atomic_write_json(manifest_path, build_manifest(args.candidates, args.claims, cards, selection, manifest_files))

    from research.phase1.microconcepts.validate_microconcept_labels_v1 import validate_paths
    validation = validate_paths(
        labels_path=paths["labels"], tasks_path=paths["tasks"], queue_path=paths["queue"],
        frozen_path=paths["frozen"], manifest_path=manifest_path, candidates_path=args.candidates,
        claims_path=args.claims, audit_size=50,
    )
    atomic_write_json(paths["validation"], validation)
    if validation["error_count"]:
        raise RuntimeError(f"v1 label validation failed with {validation['error_count']} errors")
    atomic_write_json(manifest_path, build_manifest(args.candidates, args.claims, cards, selection, [*manifest_files, paths["validation"]]))
    print(json.dumps({
        "episodes": selection["pilot_size"], "state_labels": len(labels), "task_labels": len(tasks),
        "review_queue": len(queue), "frozen_time_test": selection["frozen_time_test_size"],
        "validation_errors": validation["error_count"], "output": str(args.output),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
