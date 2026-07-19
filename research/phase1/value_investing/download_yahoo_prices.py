"""Download exploratory historical daily prices for a collected SEC universe.

This is intentionally separate from the SEC collector: ticker history and
delisting returns remain unresolved and the result cannot support a research-grade
backtest. Raw price files must remain outside the repository.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

try:
    from .download_free_pilot import external_output
    from .free_sources import YahooFinanceFreePriceClient
except ImportError:
    from download_free_pilot import external_output
    from free_sources import YahooFinanceFreePriceClient


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp())


def load_or_create_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"manifest_version": "yahoo-price-v1", "provider": "Yahoo Finance chart endpoint", "price_basis": "adjusted close", "limitations": ["current ticker is not a verified historical ticker mapping", "no delisting return coverage", "exploratory use only"], "issuers": {}}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def download(raw_root: Path, output: Path, start: str, end: str, delay_seconds: float = 0.75, client: YahooFinanceFreePriceClient | None = None) -> dict[str, Any]:
    raw_root, output = external_output(raw_root), external_output(output)
    sec_manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = load_or_create_manifest(manifest_path)
    price_client = client or YahooFinanceFreePriceClient()
    for cik, issuer in sec_manifest["issuers"].items():
        if issuer.get("status") != "accepted" or manifest["issuers"].get(cik, {}).get("status") == "downloaded":
            continue
        ticker = issuer["ticker"]
        try:
            rows = price_client.daily_adjusted_closes(ticker, epoch(start), epoch(end))
            destination = output / "prices" / f"{cik}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"cik": cik, "current_ticker": ticker, "source": "Yahoo Finance adjusted close", "start": start, "end": end, "rows": [{"date": row.trading_date, "adjusted_close": row.close} for row in rows]}, indent=2), encoding="utf-8")
            manifest["issuers"][cik] = {"status": "downloaded", "ticker": ticker, "rows": len(rows), "path": str(destination.relative_to(output))}
        except Exception as error:
            manifest["issuers"][cik] = {"status": "failed", "ticker": ticker, "error_type": type(error).__name__, "error": str(error)[:300]}
        write_manifest(manifest_path, manifest)
        time.sleep(delay_seconds)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--delay-seconds", type=float, default=0.75)
    args = parser.parse_args()
    manifest = download(args.raw_root, args.output, args.start, args.end, args.delay_seconds)
    downloaded = sum(1 for item in manifest["issuers"].values() if item["status"] == "downloaded")
    print(f"Downloaded exploratory Yahoo price histories for {downloaded} issuers to {external_output(args.output)}")


if __name__ == "__main__":
    main()
