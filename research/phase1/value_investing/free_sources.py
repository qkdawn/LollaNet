"""Free public-source adapters for the exploratory US value-investing pilot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import csv
import io
import json
import os
from typing import Any, Callable
from urllib.request import Request, urlopen
from urllib.parse import quote


SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"
STOOQ_DAILY = "https://stooq.com/q/d/l/"
ALPHA_VANTAGE = "https://www.alphavantage.co/query"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"


class FreeSourceConfigurationError(RuntimeError):
    """Raised before a public-data request lacking required source configuration."""


@dataclass(frozen=True)
class SecFact:
    tag: str
    value: float
    unit: str
    filed: str
    period_end: str
    form: str
    accession: str


@dataclass(frozen=True)
class DailyClose:
    trading_date: str
    close: float


@dataclass(frozen=True)
class MonthlyAdjustedClose:
    trading_date: str
    close: float
    adjusted_close: float


def _default_fetch(request: Request) -> bytes:
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed public HTTPS endpoints
        return response.read()


class SecEdgarFreeClient:
    """SEC EDGAR client that requires the caller's truthful contact User-Agent."""

    def __init__(self, user_agent: str | None = None, fetch: Callable[[Request], bytes] = _default_fetch) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        if "@" not in self.user_agent:
            raise FreeSourceConfigurationError(
                "Set SEC_USER_AGENT to a truthful identifier containing a contact email before requesting SEC data."
            )
        self.fetch = fetch

    def _json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": self.user_agent})
        return json.loads(self.fetch(request).decode("utf-8"))

    def company_tickers(self) -> dict[str, Any]:
        return self._json(f"{SEC_WWW}/files/company_tickers.json")

    def submissions(self, cik: str) -> dict[str, Any]:
        return self._json(f"{SEC_DATA}/submissions/CIK{int(cik):010d}.json")

    def company_facts(self, cik: str) -> dict[str, Any]:
        return self._json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{int(cik):010d}.json")


def facts_available_as_of(company_facts: dict[str, Any], tags: list[str], as_of: str) -> dict[str, SecFact]:
    """Choose the latest 10-K/10-Q fact filed on or before the decision timestamp."""
    cutoff = date.fromisoformat(as_of)
    result: dict[str, SecFact] = {}
    gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        observations = []
        for unit, rows in gaap_facts.get(tag, {}).get("units", {}).items():
            for row in rows:
                filed = row.get("filed")
                if row.get("form") not in {"10-K", "10-Q"} or not filed or date.fromisoformat(filed) > cutoff:
                    continue
                if "val" not in row or "end" not in row:
                    continue
                observations.append((filed, row["end"], unit, row))
        if observations:
            filed, period_end, unit, row = max(observations, key=lambda item: (item[0], item[1]))
            result[tag] = SecFact(tag, float(row["val"]), unit, filed, period_end, row["form"], row.get("accn", ""))
    return result


class StooqFreePriceClient:
    """Public daily-price reader for exploratory use; it is not delisting-complete."""

    def __init__(self, fetch: Callable[[Request], bytes] = _default_fetch) -> None:
        self.fetch = fetch

    def daily_closes(self, symbol: str) -> list[DailyClose]:
        normalized = symbol.lower().replace(".", "-")
        request = Request(f"{STOOQ_DAILY}?s={normalized}.us&i=d", headers={"User-Agent": "LollaNet research prototype"})
        text = self.fetch(request).decode("utf-8")
        if text.lstrip().startswith("<"):
            raise FreeSourceConfigurationError(
                "Stooq returned an anti-automation page; use Alpha Vantage or provide a licensed price source."
            )
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            if not row.get("Date") or not row.get("Close") or row["Close"] in {"N/D", ""}:
                continue
            rows.append(DailyClose(row["Date"], float(row["Close"])))
        return rows


class AlphaVantageFreePriceClient:
    """Free-tier raw monthly closes; it does not provide adjusted total-return prices."""

    def __init__(self, api_key: str | None = None, fetch: Callable[[Request], bytes] = _default_fetch) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            raise FreeSourceConfigurationError("Set ALPHAVANTAGE_API_KEY before requesting Alpha Vantage data.")
        self.fetch = fetch

    def monthly_closes(self, symbol: str) -> list[DailyClose]:
        url = f"{ALPHA_VANTAGE}?function=TIME_SERIES_MONTHLY&symbol={symbol.upper()}&apikey={self.api_key}"
        payload = json.loads(self.fetch(Request(url, headers={"User-Agent": "LollaNet research prototype"})).decode("utf-8"))
        if "Note" in payload or "Information" in payload or "Error Message" in payload:
            raise FreeSourceConfigurationError(f"Alpha Vantage request failed: {payload.get('Note') or payload.get('Information') or payload.get('Error Message')}")
        rows = payload.get("Monthly Time Series")
        if not isinstance(rows, dict):
            raise FreeSourceConfigurationError("Alpha Vantage response did not contain monthly prices.")
        return [DailyClose(day, float(values["4. close"])) for day, values in sorted(rows.items())]

    def monthly_adjusted_closes(self, symbol: str) -> list[MonthlyAdjustedClose]:
        url = f"{ALPHA_VANTAGE}?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol.upper()}&apikey={self.api_key}"
        payload = json.loads(self.fetch(Request(url, headers={"User-Agent": "LollaNet research prototype"})).decode("utf-8"))
        if "Note" in payload or "Information" in payload or "Error Message" in payload:
            raise FreeSourceConfigurationError(f"Alpha Vantage request failed: {payload.get('Note') or payload.get('Information') or payload.get('Error Message')}")
        rows = payload.get("Monthly Adjusted Time Series")
        if not isinstance(rows, dict):
            raise FreeSourceConfigurationError("Alpha Vantage response did not contain monthly adjusted prices.")
        return [
            MonthlyAdjustedClose(day, float(values["4. close"]), float(values["5. adjusted close"]))
            for day, values in sorted(rows.items())
        ]


class YahooFinanceFreePriceClient:
    """Unofficial public daily prices for exploratory feature completion only.

    Yahoo's chart endpoint returns adjusted closes, but it has no delisting-return
    contract and current tickers are not a historical CIK-to-security mapping.
    """

    def __init__(self, fetch: Callable[[Request], bytes] = _default_fetch) -> None:
        self.fetch = fetch

    def daily_adjusted_closes(self, symbol: str, start_epoch: int, end_epoch: int) -> list[DailyClose]:
        url = f"{YAHOO_CHART}/{quote(symbol.upper(), safe='.-')}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=history"
        payload = json.loads(self.fetch(Request(url, headers={"User-Agent": "Mozilla/5.0 LollaNet research prototype"})).decode("utf-8"))
        result = payload.get("chart", {}).get("result")
        if not result:
            message = payload.get("chart", {}).get("error", {}).get("description", "unknown Yahoo Finance response")
            raise FreeSourceConfigurationError(f"Yahoo Finance request failed: {message}")
        series = result[0]
        timestamps = series.get("timestamp", [])
        adjusted = series.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        return [DailyClose(datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(), float(close)) for timestamp, close in zip(timestamps, adjusted) if close is not None]
