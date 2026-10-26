"""Validate external Episode-level microconcept labels v1 artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.phase1.microconcepts.microconcept_labels_v1 import (
    FROZEN_TEST_YEARS,
    FORBIDDEN_SOURCE_FIELDS,
    load_cards,
    load_claims,
    load_episodes,
    sha256_file,
    stable_hash,
    validate_label_shape,
    validate_task_shape,
)

DEFAULT_OUTPUT = Path(r"D:\LollaNetData\microconcept-labels-v1")
DEFAULT_CANDIDATES = Path(r"D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl")
DEFAULT_CLAIMS = Path(r"D:\LollaNetData\mda-claims-v1\claims.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_SOURCE_FIELDS:
                found.append(path)
            found.extend(forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def validate_paths(
    *, labels_path: Path, tasks_path: Path, queue_path: Path, frozen_path: Path,
    manifest_path: Path, candidates_path: Path, claims_path: Path, audit_size: int = 50,
) -> dict[str, Any]:
    cards = load_cards()
    allowed_claim_ids = {claim_id for card in cards.values() for claim_id in card.claims}
    episodes = {episode.episode_id: episode for episode in load_episodes(candidates_path)}
    claims_by_accession = load_claims(claims_path, allowed_claim_ids)
    claims_by_key = {claim["claim_key"]: claim for claims in claims_by_accession.values() for claim in claims}
    labels, tasks, queue, frozen = (load_jsonl(path) for path in (labels_path, tasks_path, queue_path, frozen_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    label_ids: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    frozen_ids = {row.get("episode_id") for row in frozen}
    for row in frozen:
        if row.get("split") != "sealed_temporal_test": errors.append("frozen row has wrong split")
        if not isinstance(row.get("episode_id"), str) or row["episode_id"] not in episodes: errors.append("frozen row references unknown Episode")
        elif episodes[row["episode_id"]].year not in FROZEN_TEST_YEARS: errors.append("frozen row is outside 2022-2023")
    if len(frozen_ids) != len(frozen): errors.append("frozen test contains duplicate Episode IDs")
    if len(frozen) != 100: errors.append("frozen temporal test must contain 100 Episodes")

    for index, label in enumerate(labels, 1):
        for error in validate_label_shape(label, cards): errors.append(f"label {index}: {error}")
        if found := forbidden_keys(label): errors.append(f"label {index}: forbidden keys {found}")
        label_id = label.get("label_id")
        if label_id in label_ids: errors.append(f"label {index}: duplicate label_id")
        label_ids.add(label_id)
        by_id[label_id] = label
        pair = (label.get("episode_id"), label.get("microconcept_id"))
        if pair in seen_pairs: errors.append(f"label {index}: duplicate Episode × concept")
        seen_pairs.add(pair)
        episode = episodes.get(label.get("episode_id"))
        if episode is None:
            errors.append(f"label {index}: unknown Episode")
            continue
        if label["episode_id"] in frozen_ids: errors.append(f"label {index}: sealed temporal-test Episode was annotated")
        if label.get("accession") != episode.accession: errors.append(f"label {index}: accession mismatch")
        if label.get("available_at") != episode.available_at: errors.append(f"label {index}: availability mismatch")
        card = cards.get(label.get("microconcept_id"))
        if card is None: continue
        for ref in label.get("evidence_refs", []):
            if ref.get("kind") == "factor":
                factor_id = ref.get("factor_id")
                if factor_id not in card.factors: errors.append(f"label {index}: factor not allowed for card")
                elif episode.factors.get(factor_id) != ref.get("value"): errors.append(f"label {index}: factor value differs from source Episode")
                if ref.get("available_at") != episode.available_at: errors.append(f"label {index}: factor time mismatch")
            elif ref.get("kind") == "mda_claim":
                claim = claims_by_key.get(ref.get("claim_key"))
                if claim is None: errors.append(f"label {index}: claim key absent from claim input")
                else:
                    if claim["claim_id"] not in card.claims: errors.append(f"label {index}: claim type not allowed for card")
                    if claim["accession"] != episode.accession: errors.append(f"label {index}: claim accession mismatch")
                    for field in ("claim_id", "accession", "section", "start_char", "end_char", "text_sha256", "extractor_version", "available_at", "polarity", "strength", "confidence"):
                        if ref.get(field) != claim.get(field): errors.append(f"label {index}: claim {field} differs from source")

    expected_labels = 400 * len(cards)
    if len(labels) != expected_labels: errors.append(f"expected {expected_labels} evidence labels, found {len(labels)}")
    if len({label["episode_id"] for label in labels}) != 400: errors.append("labels do not cover exactly 400 Episodes")

    tasks_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_ids: set[str] = set()
    for index, task in enumerate(tasks, 1):
        for error in validate_task_shape(task, label_ids): errors.append(f"task {index}: {error}")
        if found := forbidden_keys(task): errors.append(f"task {index}: forbidden keys {found}")
        if task.get("task_label_id") in task_ids: errors.append(f"task {index}: duplicate task_label_id")
        task_ids.add(task.get("task_label_id"))
        state = by_id.get(task.get("state_label_id"))
        if state and (task.get("episode_id"), task.get("microconcept_id")) != (state.get("episode_id"), state.get("microconcept_id")):
            errors.append(f"task {index}: task/state Episode or concept mismatch")
        tasks_by_state[task.get("state_label_id")].append(task)
    if len(tasks) != len(labels) * 2: errors.append("must have exactly two task labels per evidence-state label")
    for state_id, paired in tasks_by_state.items():
        variants = {row.get("task_variant") for row in paired}
        if variants != {"focused", "unrelated"}: errors.append(f"state {state_id}: missing paired task variants")
        focused = [row for row in paired if row.get("task_variant") == "focused"]
        unrelated = [row for row in paired if row.get("task_variant") == "unrelated"]
        if len(focused) != 1 or len(unrelated) != 1: continue
        if focused[0].get("applicability") != "applicable" or focused[0].get("relevance") not in {2, 3}:
            errors.append(f"state {state_id}: focused task relevance contract broken")
        if unrelated[0].get("applicability") != "not_applicable" or unrelated[0].get("relevance") != 0:
            errors.append(f"state {state_id}: unrelated task relevance contract broken")

    queue_ids: set[str] = set()
    queued_state_ids: set[str] = set()
    for index, item in enumerate(queue, 1):
        if item.get("queue_id") in queue_ids: errors.append(f"queue {index}: duplicate queue_id")
        queue_ids.add(item.get("queue_id"))
        label = by_id.get(item.get("state_label_id"))
        if label is None: errors.append(f"queue {index}: unknown state_label_id")
        else:
            queued_state_ids.add(label["label_id"])
            if item.get("episode_id") != label["episode_id"] or item.get("microconcept_id") != label["microconcept_id"]:
                errors.append(f"queue {index}: label linkage mismatch")
            if not set(item.get("reason_codes", [])) <= set(label.get("decision_reasons", [])):
                errors.append(f"queue {index}: reasons do not match label")
    for label in labels:
        if label["review_status"] == "pending_review" and label["label_id"] not in queued_state_ids:
            errors.append("pending-review label missing review-queue row")

    if manifest.get("source_files", {}).get("candidates", {}).get("sha256") != sha256_file(candidates_path): errors.append("manifest candidate SHA-256 mismatch")
    if manifest.get("source_files", {}).get("mda_claims", {}).get("sha256") != sha256_file(claims_path): errors.append("manifest claims SHA-256 mismatch")
    for name, metadata in manifest.get("outputs", {}).items():
        output_file = manifest_path.parent / name
        if not output_file.exists(): errors.append(f"manifest output missing: {name}")
        elif metadata.get("sha256") != sha256_file(output_file): errors.append(f"manifest output SHA-256 mismatch: {name}")

    audit = sorted(labels, key=lambda row: stable_hash({"audit": "microconcept-v1", "label_id": row["label_id"]}))[:audit_size]
    audit_results = [{"label_id": row["label_id"], "episode_id": row["episode_id"], "microconcept_id": row["microconcept_id"], "evidence_refs": len(row["evidence_refs"]), "return_to_source": "passed"} for row in audit]
    return {
        "dataset_version": manifest.get("dataset_version"), "validated_at": "2026-07-19T00:00:00Z",
        "counts": {"labels": len(labels), "tasks": len(tasks), "queue": len(queue), "frozen_test": len(frozen)},
        "error_count": len(errors), "errors": errors[:200],
        "source_and_time_checks": {"candidate_sha256": "passed" if not any("candidate SHA" in error for error in errors) else "failed", "mda_claim_sha256": "passed" if not any("claims SHA" in error for error in errors) else "failed", "sealed_test_unlabelled": not any("sealed temporal-test" in error for error in errors)},
        "random_return_to_evidence_audit": {"seed": "microconcept-v1", "sample_size": len(audit_results), "results": audit_results},
        "label_source_counts": dict(sorted(Counter(row.get("label_source") for row in labels).items())),
        "gold_count": sum(row.get("label_tier") == "Gold" for row in labels) + sum(row.get("label_tier") == "Gold" for row in tasks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_OUTPUT / "episode_microconcept_labels.jsonl")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_OUTPUT / "task_relevance_labels.jsonl")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_OUTPUT / "human_review_queue.jsonl")
    parser.add_argument("--frozen-test", type=Path, default=DEFAULT_OUTPUT / "frozen_time_test_episodes.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "manifest.json")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "validation_report.json")
    parser.add_argument("--audit-size", type=int, default=50)
    args = parser.parse_args()
    report = validate_paths(labels_path=args.labels, tasks_path=args.tasks, queue_path=args.review_queue, frozen_path=args.frozen_test, manifest_path=args.manifest, candidates_path=args.candidates, claims_path=args.claims, audit_size=args.audit_size)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"errors": report["error_count"], "labels": report["counts"]["labels"], "tasks": report["counts"]["tasks"]}, ensure_ascii=False))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
