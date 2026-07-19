"""Attach clearly limited free-price features to SEC financial candidates."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

try:
    from .download_free_pilot import external_output
except ImportError:
    from download_free_pilot import external_output


def day(value: str) -> date:
    return date.fromisoformat(value[:10])


def close_after(rows: list[dict[str, Any]], target: date) -> tuple[int, float] | None:
    dates = [row["date"] for row in rows]
    index = bisect_right(dates, target.isoformat())
    if index >= len(rows):
        return None
    return index, float(rows[index]["adjusted_close"])


def shares_outstanding(facts: dict[str, Any], filing_day: date, accession: str) -> float | None:
    rows = []
    for namespace, tag in (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("dei", "EntityCommonSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ):
        rows.extend(facts.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {}).get("shares", []))
    candidates = [row for row in rows if row.get("filed") and (row.get("accn") == accession or day(row["filed"]) < filing_day) and "val" in row]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: (row.get("end", ""), row.get("filed", "")))["val"])


def drawdown(rows: list[dict[str, Any]], entry_index: int) -> float | None:
    window = [float(row["adjusted_close"]) for row in rows[max(0, entry_index - 251):entry_index + 1]]
    if len(window) < 100:
        return None
    peak = window[0]
    worst = 0.0
    for value in window:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def enrich(candidates: Path, raw_root: Path, prices_root: Path, output: Path) -> dict[str, Any]:
    candidates, raw_root, prices_root, output = map(external_output, (candidates, raw_root, prices_root, output))
    price_manifest = json.loads((prices_root / "manifest.json").read_text(encoding="utf-8"))
    price_rows: dict[str, list[dict[str, Any]]] = {}
    for cik, item in price_manifest["issuers"].items():
        if item.get("status") == "downloaded":
            payload = json.loads((prices_root / item["path"]).read_text(encoding="utf-8"))
            price_rows[cik] = payload["rows"]
    facts_cache: dict[str, dict[str, Any]] = {}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "sec_candidates_market_enriched.jsonl"
    stats = {"total": 0, "price_entry_available": 0, "market_cap_available": 0, "drawdown_available": 0, "price_outcome_available": 0}
    with candidates.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            stats["total"] += 1
            cik = record["cik"]
            rows = price_rows.get(cik, [])
            availability = day(record["availability"]["public_available_at"])
            entry = close_after(rows, availability)
            market = record["market_state"]
            market["sector_code"] = f"SIC-{str(record.get('sic', '') or 'unknown')[:2]}" if record.get("sic") else None
            market["market_cap"] = market["free_cash_flow_yield"] = market["trailing_drawdown"] = None
            outcome = {"status": "unavailable"}
            if entry:
                entry_index, entry_close = entry
                stats["price_entry_available"] += 1
                if cik not in facts_cache:
                    facts_cache[cik] = json.loads((raw_root / "issuers" / cik / "company_facts.json").read_text(encoding="utf-8"))
                shares = shares_outstanding(facts_cache[cik], availability, record["filing"]["accession"])
                if shares is not None:
                    market["market_cap"] = entry_close * shares
                    stats["market_cap_available"] += 1
                    fcf = record["financial_features"].get("ttm_free_cash_flow")
                    if fcf is not None and market["market_cap"]:
                        market["free_cash_flow_yield"] = fcf / market["market_cap"]
                market["trailing_drawdown"] = drawdown(rows, entry_index)
                if market["trailing_drawdown"] is not None:
                    stats["drawdown_available"] += 1
                exit_quote = close_after(rows, availability + timedelta(days=365))
                if exit_quote:
                    _exit_index, exit_close = exit_quote
                    outcome = {"status": "exploratory_price_only", "entry_date": rows[entry_index]["date"], "entry_adjusted_close": entry_close, "exit_date": rows[_exit_index]["date"], "exit_adjusted_close": exit_close, "raw_price_return": exit_close / entry_close - 1, "delisting_return_included": False}
                    stats["price_outcome_available"] += 1
            record["market_provenance"] = {"provider": "Yahoo Finance chart endpoint", "record_version": "yahoo-price-v1", "historical_ticker_mapping": "unverified_current_ticker", "delisting_return_coverage": False}
            record["exploratory_outcome"] = outcome
            record["training_status"] = "provisional_free_price_current_ticker_mapping"
            target.write(json.dumps(record, sort_keys=True) + "\n")
    report = {"record_type": "exploratory_market_enrichment_v1", **stats, "limitations": ["current ticker is not a verified historical CIK-to-security mapping", "Yahoo adjusted price does not provide complete delisting returns", "valuation percentile remains unavailable until a point-in-time peer universe is established"]}
    (output / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--prices-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = enrich(args.candidates, args.raw_root, args.prices_root, args.output)
    print(f"Enriched {report['total']} candidates; entry prices available for {report['price_entry_available']}")


if __name__ == "__main__":
    main()
