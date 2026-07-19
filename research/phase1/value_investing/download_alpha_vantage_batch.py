"""Resumable, rate-limited download of exploratory monthly adjusted prices."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

try:
    from .download_free_pilot import external_output
    from .free_sources import AlphaVantageFreePriceClient
except ImportError:
    from download_free_pilot import external_output
    from free_sources import AlphaVantageFreePriceClient


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"provider": "Alpha Vantage TIME_SERIES_MONTHLY_ADJUSTED", "record_version": "alpha-vantage-monthly-adjusted-v1", "prices": {}}


def _save(path: Path, value: dict[str, Any]) -> None:
    value["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def download(raw_root: Path, output: Path, limit: int, requests_per_minute: float = 5, api_key: str | None = None) -> dict[str, Any]:
    """Download up to ``limit`` uncollected ticker histories; resume safely later."""
    if limit < 1 or requests_per_minute <= 0:
        raise ValueError("limit and requests_per_minute must be positive")
    raw_root, output = external_output(raw_root), external_output(output)
    issuer_manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = _load(manifest_path)
    client = AlphaVantageFreePriceClient(api_key)
    interval = 60 / requests_per_minute
    selected = [
        (cik, issuer) for cik, issuer in issuer_manifest["issuers"].items()
        if issuer.get("status") == "accepted" and manifest["prices"].get(cik, {}).get("status") != "downloaded"
    ][:limit]
    for position, (cik, issuer) in enumerate(selected):
        ticker = issuer["ticker"]
        record = {"ticker_at_collection": ticker, "status": "pending", "updated_at": datetime.now(UTC).isoformat()}
        manifest["prices"][cik] = record
        try:
            prices = client.monthly_adjusted_closes(ticker)
            filename = f"{cik}_{ticker.upper()}.json"
            (output / filename).write_text(json.dumps([asdict(price) for price in prices], indent=2), encoding="utf-8")
            record.update({"status": "downloaded", "path": filename, "observations": len(prices), "first_date": prices[0].trading_date, "last_date": prices[-1].trading_date})
        except Exception as error:
            record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)[:300]})
        finally:
            record["updated_at"] = datetime.now(UTC).isoformat()
            _save(manifest_path, manifest)
        if position + 1 < len(selected):
            time.sleep(interval)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20, help="Maximum new symbols this run; keep within provider quota")
    parser.add_argument("--requests-per-minute", type=float, default=5)
    parser.add_argument("--alphavantage-api-key", help="Defaults to ALPHAVANTAGE_API_KEY")
    args = parser.parse_args()
    manifest = download(args.raw_root, args.output, args.limit, args.requests_per_minute, args.alphavantage_api_key)
    complete = sum(1 for value in manifest["prices"].values() if value.get("status") == "downloaded")
    print(f"Downloaded monthly adjusted prices for {complete} issuers to {external_output(args.output)}")


if __name__ == "__main__":
    main()
