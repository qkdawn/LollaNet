"""Validate schema, source bounds, hashes, time availability, and auditability for MD&A claims v1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.phase1.value_investing.download_sec_mda import normalize_html
from research.phase1.value_investing.mda_claims_v1 import (
    FORBIDDEN_INPUT_OR_OUTPUT_KEYS,
    atomic_write_json,
    claims_distribution,
    load_claim_types,
    sha256_text,
    stable_hash,
    validate_claim_shape,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") == "located" and item.get("locator_quality") == "high":
                index[item["accession"]] = item
    return index


def validate(args: argparse.Namespace) -> dict:
    claim_types = load_claim_types(args.taxonomy)
    claims = read_jsonl(args.claims)
    index = load_index(args.index)
    errors: list[dict] = []
    by_accession: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    seen_keys: set[str] = set()
    for number, claim in enumerate(claims, 1):
        for error in validate_claim_shape(claim, claim_types):
            errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": error})
        if claim.get("claim_key") in seen_keys:
            errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "duplicate claim_key"})
        seen_keys.add(claim.get("claim_key"))
        by_accession[claim.get("accession")].append((number, claim))

    validated = 0
    for accession, rows in sorted(by_accession.items()):
        source = index.get(accession)
        if source is None:
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "accession absent from high-confidence MD&A index"})
            continue
        availability = source.get("available_at")
        if not isinstance(availability, str) or not availability.endswith("Z"):
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "source lacks point-in-time public availability timestamp"})
            continue
        raw_path = Path(source["raw_path"])
        try:
            raw = raw_path.read_bytes()
        except OSError as exc:
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": f"raw filing unreadable: {exc}"})
            continue
        if hashlib.sha256(raw).hexdigest() != source["raw_html_sha256"]:
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "raw_html_sha256 mismatch"})
            continue
        text = normalize_html(raw)
        start, end = int(source["start_char"]), int(source["end_char"])
        if not (0 <= start < end <= len(text)):
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "source locator bounds invalid"})
            continue
        section = text[start:end]
        if sha256_text(section) != source["text_sha256"]:
            for number, claim in rows:
                errors.append({"line": number, "claim_key": claim.get("claim_key"), "error": "source locator text_sha256 mismatch"})
            continue
        for number, claim in rows:
            if claim["available_at"] != availability:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "available_at differs from source locator"})
            if claim["section"] != source["section"]:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "section differs from locator"})
            if claim["text_sha256"] != source["text_sha256"]:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "text_sha256 differs from locator"})
            if claim["extractor_version"] != source["extractor_version"]:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "extractor_version differs from locator"})
            claim_start, claim_end = claim["start_char"], claim["end_char"]
            if not (start <= claim_start < claim_end <= end):
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "claim bounds lie outside locator section"})
                continue
            if sha256_text(text[claim_start:claim_end]) != claim["evidence_sha256"]:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "evidence_sha256 mismatch"})
                continue
            expected_key = stable_hash({"accession": accession, "claim_id": claim["claim_id"], "start_char": claim_start, "end_char": claim_end, "evidence_sha256": claim["evidence_sha256"]})
            if expected_key != claim["claim_key"]:
                errors.append({"line": number, "claim_key": claim["claim_key"], "error": "claim_key is not deterministic from evidence"})
                continue
            validated += 1

    audit_count = min(args.audit_size, len(claims))
    audit_claims = sorted(claims, key=lambda item: stable_hash({"seed": args.audit_seed, "claim_key": item["claim_key"]}))[:audit_count]
    error_keys = {error.get("claim_key") for error in errors}
    audit = [{"claim_key": item["claim_key"], "accession": item["accession"], "section": item["section"], "start_char": item["start_char"], "end_char": item["end_char"], "evidence_sha256": item["evidence_sha256"], "passed": item["claim_key"] not in error_keys} for item in audit_claims]
    report = {
        "record_type": "mda_claims_validation_report_v1",
        "claims_read": len(claims),
        "claims_validated_against_source": validated,
        "error_count": len(errors),
        "error_type_distribution": dict(sorted(Counter(item["error"] for item in errors).items())),
        "errors": errors,
        "label_distribution": claims_distribution(claims),
        "random_audit": {"seed": args.audit_seed, "requested": args.audit_size, "completed": audit_count, "all_passed": all(item["passed"] for item in audit), "claims": audit},
        "data_leakage_boundary": {"forbidden_fields": sorted(FORBIDDEN_INPUT_OR_OUTPUT_KEYS), "status": "passed" if not errors else "failed"},
    }
    atomic_write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", default=Path(r"D:\\LollaNetData\\mda-claims-v1\\claims.jsonl"), type=Path)
    parser.add_argument("--index", default=Path(r"D:\\LollaNetData\\sec-mda-v1\\mda_index.jsonl"), type=Path)
    parser.add_argument("--taxonomy", default=Path("research/phase1/microconcepts/qualitative_claim_types.yaml"), type=Path)
    parser.add_argument("--output", default=Path(r"D:\\LollaNetData\\mda-claims-v1\\validation_report.json"), type=Path)
    parser.add_argument("--audit-size", type=int, default=25)
    parser.add_argument("--audit-seed", default="mda-claims-v1-audit-20260719")
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps({key: report[key] for key in ("claims_read", "claims_validated_against_source", "error_count", "random_audit")}, ensure_ascii=False, indent=2, sort_keys=True))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
