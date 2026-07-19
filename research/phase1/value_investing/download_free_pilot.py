"""Download a small free-data pilot outside the repository for one US issuer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .free_sources import AlphaVantageFreePriceClient, SecEdgarFreeClient, StooqFreePriceClient
except ImportError:  # Supports direct execution from the repository root.
    from free_sources import AlphaVantageFreePriceClient, SecEdgarFreeClient, StooqFreePriceClient


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]


def external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError("output directory must be outside the repository; do not commit downloaded data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", required=True, help="SEC Central Index Key, with or without leading zeros")
    parser.add_argument("--output", required=True, type=Path, help="External directory for downloaded raw data")
    parser.add_argument("--ticker", help="Optional US ticker for exploratory daily closes")
    parser.add_argument("--price-source", choices=("stooq", "alpha_vantage"), default="stooq")
    parser.add_argument("--alphavantage-api-key", help="Free Alpha Vantage API key; defaults to ALPHAVANTAGE_API_KEY")
    parser.add_argument("--sec-user-agent", help="Truthful SEC User-Agent containing a contact email; defaults to SEC_USER_AGENT")
    args = parser.parse_args()

    output = external_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sec = SecEdgarFreeClient(args.sec_user_agent)
    (output / "submissions.json").write_text(json.dumps(sec.submissions(args.cik), indent=2), encoding="utf-8")
    (output / "company_facts.json").write_text(json.dumps(sec.company_facts(args.cik), indent=2), encoding="utf-8")
    if args.ticker:
        client = (
            AlphaVantageFreePriceClient(args.alphavantage_api_key)
            if args.price_source == "alpha_vantage"
            else StooqFreePriceClient()
        )
        closes = (
            client.monthly_closes(args.ticker)
            if args.price_source == "alpha_vantage"
            else client.daily_closes(args.ticker)
        )
        interval = "monthly" if args.price_source == "alpha_vantage" else "daily"
        (output / f"{args.price_source}_{interval}_closes.json").write_text(
            json.dumps([{"date": item.trading_date, "close": item.close} for item in closes], indent=2),
            encoding="utf-8",
        )
    print(f"Downloaded free pilot inputs to {output}")


if __name__ == "__main__":
    main()
