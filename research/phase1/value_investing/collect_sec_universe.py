"""Resumably collect a non-financial SEC filing universe outside the repository.

This downloader intentionally collects only SEC metadata and Company Facts.  It
does not fetch prices, calculate returns, or claim that an issuer is investable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

try:
    from .download_free_pilot import external_output
    from .free_sources import SecEdgarFreeClient
except ImportError:  # Supports direct execution from the repository root.
    from download_free_pilot import external_output
    from free_sources import SecEdgarFreeClient


MANIFEST_VERSION = "sec-universe-v1"
FINANCIAL_SIC_MIN = 6000
FINANCIAL_SIC_MAX = 6799
FILING_FORMS = {"10-K", "10-Q"}


@dataclass(frozen=True)
class Candidate:
    cik: str
    ticker: str
    title: str


class RequestPacer:
    """Keep requests below the SEC's published ten-requests-per-second ceiling."""

    def __init__(self, requests_per_second: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval = 1 / requests_per_second
        self.clock = clock
        self.sleep = sleep
        self.last_request_at: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.last_request_at is not None:
            remaining = self.interval - (now - self.last_request_at)
            if remaining > 0:
                self.sleep(remaining)
        self.last_request_at = self.clock()


def canonical_cik(value: str | int) -> str:
    return f"{int(value):010d}"


def select_candidates(payload: dict[str, Any], target: int, seed: int) -> list[Candidate]:
    """Choose a reproducible, deduplicated candidate list before SIC screening."""
    values = payload.values() if all(isinstance(value, dict) for value in payload.values()) else []
    candidates = {
        canonical_cik(value["cik_str"]): Candidate(canonical_cik(value["cik_str"]), str(value.get("ticker", "")), str(value.get("title", "")))
        for value in values
        if value.get("cik_str") and value.get("ticker")
    }
    ordered = list(candidates.values())
    random.Random(seed).shuffle(ordered)
    # SIC is only known after submissions are requested.  A larger candidate pool
    # lets financial issuers be excluded without changing the random seed.
    return ordered[: max(target * 4, target)]


def recent_filing_count(submissions: dict[str, Any]) -> int:
    recent = submissions.get("filings", {}).get("recent", {})
    return sum(1 for form in recent.get("form", []) if form in FILING_FORMS)


def is_nonfinancial(submissions: dict[str, Any]) -> tuple[bool, str]:
    try:
        sic = int(submissions.get("sic", ""))
    except (TypeError, ValueError):
        return False, "missing_or_invalid_sic"
    if FINANCIAL_SIC_MIN <= sic <= FINANCIAL_SIC_MAX:
        return False, "financial_sic"
    if recent_filing_count(submissions) < 2:
        return False, "insufficient_10k_10q_history"
    return True, "accepted"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_version": MANIFEST_VERSION, "created_at": datetime.now(UTC).isoformat(), "issuers": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("manifest version is not compatible with this collector")
    return payload


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def request_with_retries(operation: Callable[[], dict[str, Any]], pacer: RequestPacer, attempts: int = 4) -> dict[str, Any]:
    """Retry transient SEC transport failures without increasing the request rate."""
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            pacer.wait()
            return operation()
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if index + 1 < attempts:
                time.sleep(min(2**index, 8))
    assert last_error is not None
    raise last_error


def collect(target: int, output: Path, seed: int, sec_user_agent: str | None = None, client: SecEdgarFreeClient | None = None, pacer: RequestPacer | None = None) -> dict[str, Any]:
    """Collect up to ``target`` accepted issuers and return the updated manifest."""
    if target < 1:
        raise ValueError("target must be positive")
    output = external_output(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest.update({"target_issuers": target, "selection_seed": seed, "source": "SEC EDGAR", "raw_data_policy": "external_only"})
    sec = client or SecEdgarFreeClient(sec_user_agent)
    pacer = pacer or RequestPacer(8.0)

    cached_candidates = manifest.get("candidate_pool")
    if cached_candidates:
        candidates = [Candidate(**item) for item in cached_candidates]
    else:
        candidates = select_candidates(request_with_retries(sec.company_tickers, pacer), target, seed)
        manifest["candidate_pool"] = [asdict(item) for item in candidates]
        save_manifest(manifest_path, manifest)
    issuers: dict[str, Any] = manifest["issuers"]
    accepted = sum(1 for item in issuers.values() if item.get("status") == "accepted")
    for candidate in candidates:
        if accepted >= target:
            break
        existing = issuers.get(candidate.cik)
        if existing and existing.get("status") in {"accepted", "rejected"}:
            continue
        record = {**asdict(candidate), "status": "pending", "updated_at": datetime.now(UTC).isoformat()}
        issuers[candidate.cik] = record
        try:
            submissions = request_with_retries(lambda: sec.submissions(candidate.cik), pacer)
            eligible, reason = is_nonfinancial(submissions)
            record.update({"sic": submissions.get("sic"), "sic_description": submissions.get("sicDescription"), "filing_count": recent_filing_count(submissions), "selection_reason": reason})
            if not eligible:
                record["status"] = "rejected"
                save_manifest(manifest_path, manifest)
                continue
            issuer_dir = output / "issuers" / candidate.cik
            issuer_dir.mkdir(parents=True, exist_ok=True)
            (issuer_dir / "submissions.json").write_text(json.dumps(submissions, indent=2), encoding="utf-8")
            facts = request_with_retries(lambda: sec.company_facts(candidate.cik), pacer)
            (issuer_dir / "company_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
            record.update({"status": "accepted", "downloaded_at": datetime.now(UTC).isoformat(), "paths": {"submissions": str((issuer_dir / "submissions.json").relative_to(output)), "company_facts": str((issuer_dir / "company_facts.json").relative_to(output))}})
            accepted += 1
        except Exception as error:  # Preserve error state so a later run can retry it.
            record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)[:300]})
        finally:
            record["updated_at"] = datetime.now(UTC).isoformat()
            save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200, help="Number of accepted non-financial issuers to collect")
    parser.add_argument("--output", required=True, type=Path, help="External directory for raw SEC data")
    parser.add_argument("--seed", type=int, default=20260719, help="Deterministic universe-selection seed")
    parser.add_argument("--sec-user-agent", help="Truthful SEC User-Agent with contact email; defaults to SEC_USER_AGENT")
    args = parser.parse_args()
    manifest = collect(args.target, args.output, args.seed, args.sec_user_agent)
    accepted = sum(1 for item in manifest["issuers"].values() if item.get("status") == "accepted")
    failed = sum(1 for item in manifest["issuers"].values() if item.get("status") == "failed")
    print(f"SEC collection has {accepted}/{args.target} accepted issuers; {failed} currently failed. Output: {external_output(args.output)}")


if __name__ == "__main__":
    main()
