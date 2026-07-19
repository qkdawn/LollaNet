"""Download SEC filing HTML and locate MD&A sections in external storage.

Raw filings never enter the repository. The output index stores only hashes,
offsets, accession metadata, and extraction status so later structured claims
can remain point-in-time auditable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.client import IncompleteRead
import hashlib
import html
import json
import gzip
import os
from pathlib import Path
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from lxml import html as lxml_html
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EXTRACTOR_VERSION = "sec-mda-locator-v1"
FORMS = {"10-K": ("ITEM 7", ("ITEM 7A", "ITEM 8")), "10-Q": ("ITEM 2", ("ITEM 3", "ITEM 4"))}


class RequestPacer:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            pause = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if pause:
            time.sleep(pause)


def request_bytes(url: str, user_agent: str, retries: int = 4, pacer: RequestPacer | None = None) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            if pacer:
                pacer.wait()
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed SEC HTTPS endpoint
                return response.read()
        except (HTTPError, URLError, TimeoutError, IncompleteRead) as exc:
            error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"SEC request failed for {url}: {error}")


def normalize_html(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<ix:header.*?</ix:header>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def locate_mda(text: str, form: str) -> dict[str, Any]:
    if form not in FORMS:
        return {"status": "unsupported_form"}
    start_label, end_labels = FORMS[form]
    heading = rf"\b{re.escape(start_label)}\b(?:\s*[.:\-])?\s+(?:management(?:'|’)?s\s+)?discussion"
    starts = [match.start() for match in re.finditer(heading, text, re.IGNORECASE)]
    candidates = []
    for start in starts:
        end_positions = []
        for label in end_labels:
            end_positions.extend(match.start() for match in re.finditer(rf"\b{re.escape(label)}\b(?:\s*[.:\-])?", text[start + len(start_label):], re.IGNORECASE))
        end = start + len(text) if not end_positions else start + min(end_positions)
        length = end - start
        context = text[start:start + 500].lower()
        score = int(bool(re.search(r"the following (?:discussion|analysis)", context))) * 4 + int("the purpose of this item" in context) * 2
        if score > 0 and 500 <= length <= 500_000:
            candidates.append((score, length, start, end, "high"))
    if not candidates:
        # Some legacy filings omit the formal title. Keep a conservative fallback
        # rather than selecting a distant Item reference from notes or the TOC.
        starts = [match.start() for match in re.finditer(rf"\b{re.escape(start_label)}\b(?:\s*[.:\-])?", text, re.IGNORECASE)]
        for start in starts:
            end_positions = []
            for label in end_labels:
                end_positions.extend(match.start() for match in re.finditer(rf"\b{re.escape(label)}\b(?:\s*[.:\-])?", text[start + len(start_label):], re.IGNORECASE))
            end = start + len(text) if not end_positions else start + min(end_positions)
            length = end - start
            if 500 <= length <= 500_000:
                candidates.append((0, length, start, end, "low"))
    if not candidates:
        return {"status": "section_not_found", "section": start_label}
    _, _, start, end, quality = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
    return {"status": "located" if quality == "high" else "located_low_confidence", "locator_quality": quality, "section": start_label, "start_char": start, "end_char": end, "text_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest(), "extractor_version": EXTRACTOR_VERSION}


def locate_mda_dom(raw: bytes, text: str, form: str) -> dict[str, Any] | None:
    """Resolve a filing's table-of-contents anchor to the actual MD&A node."""
    if form == "10-K":
        start_items, end_items = ("2", "6", "7"), ("7A", "8", "9")
    elif form == "10-Q":
        start_items, end_items = ("2",), ("3", "4")
    else:
        return None
    tree = lxml_html.fromstring(raw)
    candidates = []
    for anchor in tree.xpath("//a"):
        label = " ".join((anchor.text_content() or "").split())
        if not re.search(r"management(?:'|’|&#146;)?s?\s+discussion|MD&A", label, re.IGNORECASE):
            continue
        item_match = re.search(r"\bITEM\s*(2|6|7)\b", label, re.IGNORECASE)
        if item_match and item_match.group(1) not in start_items:
            continue
        href = anchor.get("href", "")
        fragment = href[1:] if isinstance(href, str) and href.startswith("#") else ""
        targets = tree.xpath(f"//*[@id={json.dumps(fragment)}] | //*[@name={json.dumps(fragment)}]") if fragment else []
        if not targets:
            continue
        target = targets[0]
        block = target
        while block is not None and len(" ".join((block.text_content() or "").split())) < 20:
            block = block.getparent()
        heading_node = block if block is not None else target
        heading = " ".join((heading_node.text_content() or "").split())
        if len(heading) < 20:
            heading = label
        phrase = heading[:160]
        anchor_text = label
        toc_pos = text.lower().find(anchor_text[:80].lower()) if anchor_text else -1
        start = text.lower().find(phrase[:80].lower(), max(0, toc_pos + len(anchor_text)))
        if start < 0:
            start = text.lower().find("management's discussion", max(0, toc_pos + len(anchor_text)))
        if start < 0:
            start = text.lower().find("management’s discussion", max(0, toc_pos + len(anchor_text)))
        if start < 0:
            continue
        item_before = list(re.finditer(r"\bITEM\s*(2|6|7)\b", text[max(0, start - 160):start], re.IGNORECASE))
        if item_before:
            start = max(0, start - 160) + item_before[-1].start()
        end_matches = []
        for item in end_items:
            end_matches.extend(re.finditer(rf"\bITEM\s*{re.escape(item)}\b(?:\s*[.:\-])?", text[start + 20:], re.IGNORECASE))
        end = start + 20 + min((match.start() for match in end_matches), default=len(text) - start - 20)
        length = end - start
        if 500 <= length <= 500_000:
            candidates.append((length, start, end, fragment))
    if not candidates:
        return None
    _, start, end, fragment = min(candidates, key=lambda row: row[0])
    return {"status": "located", "locator_quality": "high", "locator_method": "toc_anchor", "start_anchor": fragment, "section": "MD&A", "start_char": start, "end_char": end, "text_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest(), "extractor_version": EXTRACTOR_VERSION}


def locate_legacy_item6(text: str, form: str) -> dict[str, Any] | None:
    """Handle older 10-K/smaller-reporting-company Item 6 MD&A headings."""
    if form != "10-K":
        return None
    starts = [match.start() for match in re.finditer(r"\bITEM\s*6\b(?:\s*[.:\-])?\s+management(?:'|’)?s\s+discussion", text, re.IGNORECASE)]
    candidates = []
    for start in starts:
        ends = []
        for item in ("7A", "8", "9"):
            ends.extend(match.start() for match in re.finditer(rf"\bITEM\s*{item}\b", text[start + 20:], re.IGNORECASE))
        end = start + 20 + min(ends, default=len(text) - start - 20)
        if 500 <= end - start <= 500_000:
            candidates.append((end - start, start, end))
    if not candidates:
        return None
    _, start, end = min(candidates)
    return {"status": "located", "locator_quality": "high", "locator_method": "legacy_item6", "section": "ITEM 6", "start_char": start, "end_char": end, "text_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest(), "extractor_version": EXTRACTOR_VERSION}


def promote_semantic_locator(text: str, locator: dict[str, Any], form: str) -> dict[str, Any]:
    """Promote a text fallback only when heading, body, and boundary agree."""
    if locator.get("status") != "located_low_confidence":
        return locator
    start, end = locator["start_char"], locator["end_char"]
    section = text[start:end]
    heading = section[:500]
    allowed_items = "2" if form == "10-Q" else "(?:6|7)"
    heading_ok = bool(re.search(rf"^\s*ITEM\s*{allowed_items}\b.{{0,120}}management(?:['\u2019]s|s)?\s+discussion", heading, re.IGNORECASE))
    negative = bool(re.search(r"incorporated herein by reference|\b(?:see|refer to)\s+(?:part\s+[ivx]+,?\s+)?item\s*[267]", heading, re.IGNORECASE))
    signals = {
        name for name, pattern in {
            "results_of_operations": r"results? of operations?",
            "liquidity": r"\bliquidity\b",
            "cash_flows": r"cash flows?",
            "capital_resources": r"capital resources?",
            "critical_accounting": r"critical accounting",
            "financial_condition": r"financial conditions?",
            "revenue": r"\brevenues?\b",
        }.items() if re.search(pattern, section, re.IGNORECASE)
    }
    end_window = text[end:end + 160]
    expected_end = ("3", "4") if form == "10-Q" else ("7A", "8", "9")
    boundary_ok = any(re.search(rf"\bITEM\s*{item}\b", end_window, re.IGNORECASE) for item in expected_end)
    if heading_ok and not negative and len(section) >= 1_500 and len(signals) >= 2 and boundary_ok:
        return {**locator, "status": "located", "locator_quality": "high", "locator_method": "semantic_heading", "confidence_reasons": ["formal_mda_heading", "multiple_body_signals", "next_item_boundary"], "body_signals": sorted(signals)}
    return {**locator, "confidence_reasons": [reason for reason, passed in (("formal_mda_heading_missing", not heading_ok), ("negative_reference_signal", negative), ("section_too_short", len(section) < 1_500), ("insufficient_body_signals", len(signals) < 2), ("next_item_boundary_missing", not boundary_ok)) if passed]}


def submission_index(raw_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (raw_root / "issuers").glob("*/submissions.json"):
        cik = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        recent = payload.get("filings", {}).get("recent", {})
        for index, accession in enumerate(recent.get("accessionNumber", [])):
            form = recent.get("form", [None] * len(recent.get("accessionNumber", [])))[index]
            if form not in FORMS:
                continue
            result[(cik, accession)] = {
                "form": form,
                "primary_document": recent.get("primaryDocument", [None] * len(recent.get("accessionNumber", [])))[index],
                "filing_date": recent.get("filingDate", [None] * len(recent.get("accessionNumber", [])))[index],
            }
    return result


def download(input_path: Path, raw_root: Path, output: Path, user_agent: str, limit: int | None = None, delay: float = 0.2, workers: int = 4, reindex: bool = False, upgrade_low: bool = False, upgrade_missing: bool = False, semantic_only: bool = False) -> dict[str, Any]:
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    submissions = submission_index(raw_root)
    selected = records if limit is None else records[:limit]
    output.mkdir(parents=True, exist_ok=True)
    filing_dir = output / "filings"
    index_path = output / "mda_index.jsonl"
    existing = {}
    if index_path.exists():
        existing = {row["episode_id"]: row for row in (json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    failures = []
    pending = []
    for record in selected:
        episode_id = record["episode_id"]
        if not reindex and not upgrade_low and not semantic_only and episode_id in existing and existing[episode_id].get("status") in {"located", "located_low_confidence"}:
            continue
        if upgrade_low and existing.get(episode_id, {}).get("status") == "located":
            continue
        if upgrade_missing and existing.get(episode_id, {}).get("status") != "section_not_found":
            continue
        if semantic_only and existing.get(episode_id, {}).get("status") != "located_low_confidence":
            continue
        pending.append(record)

    pacer = RequestPacer(delay)

    def process(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        episode_id = record["episode_id"]
        filing = record["filing"]
        cik = str(record["cik"]).zfill(10)
        accession = filing["accession"]
        key = (cik, accession)
        meta = submissions.get(key)
        if not meta or not meta.get("primary_document"):
            return None, {"episode_id": episode_id, "reason": "submission_metadata_missing", "accession": accession}
        accession_path = accession.replace("-", "")
        url = f"{SEC_ARCHIVES}/{int(cik)}/{accession_path}/{meta['primary_document']}"
        destination = filing_dir / cik / f"{accession}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists() and destination.read_bytes()[:2] != b"\x1f\x8b":
                raw = destination.read_bytes()
            else:
                raw = request_bytes(url, user_agent, pacer=pacer)
                destination.write_bytes(raw)
            text = normalize_html(raw)
            locator = locate_mda(text, meta["form"]) if semantic_only else (locate_mda_dom(raw, text, meta["form"]) or locate_mda(text, meta["form"]))
            if locator.get("status") == "section_not_found":
                locator = locate_legacy_item6(text, meta["form"]) or locator
            locator = promote_semantic_locator(text, locator, meta["form"])
            row = {
                "episode_id": episode_id,
                "cik": cik,
                "accession": accession,
                "form": meta["form"],
                "filing_date": meta["filing_date"],
                "available_at": record["availability"]["public_available_at"],
                "source_url": url,
                "raw_html_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_path": str(destination),
                **locator,
            }
            return row, None
        except Exception as exc:  # keep the batch resumable
            return None, {"episode_id": episode_id, "reason": type(exc).__name__, "error": str(exc)[:300]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(process, record) for record in pending]
        for future in as_completed(futures):
            row, failure = future.result()
            if row:
                existing[row["episode_id"]] = row
            if failure:
                failures.append(failure)
    index_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in existing.values()), encoding="utf-8")
    report = {
        "record_type": "sec_mda_download_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(selected),
        "indexed": len(existing),
        "located": sum(row.get("status") in {"located", "located_low_confidence"} for row in existing.values()),
        "located_high_confidence": sum(row.get("status") == "located" for row in existing.values()),
        "located_low_confidence": sum(row.get("status") == "located_low_confidence" for row in existing.values()),
        "not_located": sum(row.get("status") == "section_not_found" for row in existing.values()),
        "failures": failures,
        "raw_storage": str(filing_dir),
        "extractor_version": EXTRACTOR_VERSION,
        "limitations": ["SEC filing HTML is public source material; no delisting or market outcome is implied", "historical submission files outside the cached recent submissions require a separate SEC submissions-file fetch"],
    }
    (output / "download_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reindex", action="store_true", help="re-run section location against cached HTML")
    parser.add_argument("--upgrade-low", action="store_true", help="re-run only low-confidence and unlocated cached HTML")
    parser.add_argument("--upgrade-missing", action="store_true", help="re-run only unlocated cached HTML")
    parser.add_argument("--semantic-only", action="store_true", help="re-run only low-confidence text candidates without DOM parsing")
    args = parser.parse_args()
    user_agent = args.user_agent or os.getenv("SEC_USER_AGENT", "")
    if "@" not in user_agent:
        parser.error("set SEC_USER_AGENT or --user-agent to a truthful identifier containing a contact email")
    print(json.dumps(download(args.input, args.raw_root, args.output, user_agent, args.limit, args.delay, args.workers, args.reindex, args.upgrade_low, args.upgrade_missing, args.semantic_only), indent=2))


if __name__ == "__main__":
    main()
