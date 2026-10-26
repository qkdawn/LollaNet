"""Build the real, evidence-bound MD&A claims v1 pilot outside the repository."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.phase1.value_investing.mda_claims_v1 import (
    CLAIM_DATASET_VERSION,
    CLAIM_EXTRACTOR_VERSION,
    DEFAULT_SEED,
    atomic_write_json,
    atomic_write_text,
    canonical_jsonl,
    claims_distribution,
    extract_episode,
    load_claim_types,
    load_episode_sic,
    load_locators,
    locator_from_sample_row,
    locator_to_sample_row,
    run_timestamp,
    source_distribution,
    stable_hash,
    stratified_sample,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def process_episode(locator, claim_types, checkpoint_path: Path, resume: bool) -> dict:
    if resume and checkpoint_path.exists():
        existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") == locator.fingerprint() and existing.get("claim_extractor_version") == CLAIM_EXTRACTOR_VERSION:
            return existing
    result = extract_episode(locator, claim_types)
    atomic_write_json(checkpoint_path, result)
    return result


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output == args.index.resolve().parent or output == args.raw_root.resolve():
        raise ValueError("claims output must be a distinct external directory")
    claim_types = load_claim_types(args.taxonomy)
    episode_sics = load_episode_sic(args.candidates)
    locators, status_counts, index_total = load_locators(args.index, episode_sics)
    if len(locators) != 5818:
        raise ValueError(f"Expected 5,818 high-confidence MD&A locators; found {len(locators)}")

    config = {
        "dataset_version": CLAIM_DATASET_VERSION,
        "claim_extractor_version": CLAIM_EXTRACTOR_VERSION,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "taxonomy_sha256": hashlib.sha256(args.taxonomy.read_bytes()).hexdigest(),
    }
    sample_path = output / "sample_index.jsonl"
    config_path = output / "sampling_config.json"
    if args.resume and sample_path.exists():
        expected = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        if expected != config:
            raise ValueError("existing sample configuration differs; use --no-resume with a new output directory")
        sample = [locator_from_sample_row(row) for row in read_jsonl(sample_path)]
    else:
        sample = stratified_sample(locators, args.sample_size, args.seed)
        atomic_write_text(sample_path, canonical_jsonl(locator_to_sample_row(item) for item in sample))
        atomic_write_json(config_path, config)
    if len(sample) != args.sample_size:
        raise ValueError(f"sample checkpoint contains {len(sample)} records, expected {args.sample_size}")

    checkpoints = output / "work" / "episodes"
    results = []
    for number, locator in enumerate(sample, 1):
        result = process_episode(locator, claim_types, checkpoints / f"{locator.accession}.json", args.resume)
        results.append((locator, result))
        if number % 25 == 0 or number == len(sample):
            print(f"processed {number}/{len(sample)} MD&A locators")

    claims = sorted(
        (claim for _, result in results for claim in result["claims"]),
        key=lambda item: (item["accession"], item["start_char"], item["claim_id"], item["claim_key"]),
    )
    queue = sorted(
        (item for _, result in results for item in result["review_items"]),
        key=lambda item: (item["accession"], item.get("start_char", -1), item["reason"]),
    )
    atomic_write_text(output / "claims.jsonl", canonical_jsonl(claims))
    atomic_write_text(output / "review_queue.jsonl", canonical_jsonl(queue))

    errors = Counter(item["reason"] for item in queue)
    manifest = {
        "record_type": "mda_claims_manifest_v1",
        "dataset_version": CLAIM_DATASET_VERSION,
        "claim_extractor_version": CLAIM_EXTRACTOR_VERSION,
        "created_at": run_timestamp(),
        "sampling": {**config, "sample_fingerprint": stable_hash([locator_to_sample_row(item) for item in sample])},
        "sources": {
            "candidates": str(args.candidates),
            "mda_index": str(args.index),
            "raw_root": str(args.raw_root),
            "taxonomy": str(args.taxonomy),
            "raw_text_copied": False,
            "forbidden_inputs": ["route_labels", "future_outcomes", "market_state", "financial_features", "issuer_identity"],
        },
        "input_coverage": {
            "episode_count": len(episode_sics),
            "mda_index_count": index_total,
            "high_confidence_locator_count": len(locators),
            "locator_status_counts": {
                f"{status}|{quality or 'none'}": count
                for (status, quality), count in sorted(status_counts.items())
            },
        },
        "sample": {"locator_count": len(sample), "distribution": source_distribution(sample)},
        "outputs": {
            "claim_count": len(claims),
            "review_queue_count": len(queue),
            "abstained_locator_count": sum(1 for item in queue if item["disposition"] == "abstain"),
            "label_distribution": claims_distribution(claims),
            "queue_reason_distribution": dict(sorted(errors.items())),
        },
    }
    atomic_write_json(output / "manifest.json", manifest)
    notes = """# MD&A Claims v1 Pilot Notes

## Data source and boundary

Labels were extracted only from the MD&A ranges selected by the existing SEC locator in the local SEC HTML cache. Each retained label points to a normalized-text interval and carries both the locator section hash and the interval hash. The source filing HTML remains in D:\\LollaNetData\\sec-mda-v1\\filings; this run does not copy raw MD&A text into the repository or into the label files.

The input sampler uses only filing availability time, 10-K/10-Q form, and SIC division. It does not consume issuer names, route labels, financial features, market state, or future outcomes.

## Label interpretation

claims.jsonl contains only controlled claim IDs defined in research/phase1/microconcepts/qualitative_claim_types.yaml. Labels are deterministic pseudo-labels from explicit sentence-level language, not human gold labels or investment conclusions. unknown is retained when a permitted attribute cannot be grounded conservatively. An accession with no sufficiently explicit controlled claim is represented in review_queue.jsonl as abstain, rather than by inventing a claim.

## Review queue and limitations

The review queue contains source-integrity failures, unlocatable evidence, low-confidence mappings, and abstentions. Sentence splitting is deliberately conservative and can miss claims expressed across table cells, lists, or multiple sentences. The lexical rules prioritize traceability over recall, do not resolve all negation or issuer-specific terminology, and should be human-reviewed before any downstream training. The labels must not be used as future-return, routing, or investment-result inputs.
"""
    atomic_write_text(output / "DATASET_NOTES.md", notes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--taxonomy", default=Path("research/phase1/microconcepts/qualitative_claim_types.yaml"), type=Path)
    parser.add_argument("--output", default=Path(r"D:\\LollaNetData\\mda-claims-v1"), type=Path)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
