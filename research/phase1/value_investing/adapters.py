"""Provider interfaces; real ingestion requires separately licensed access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class DataAccessUnavailable(RuntimeError):
    """Raised when a real-data adapter is invoked without licensed configuration."""


@dataclass(frozen=True)
class FilingRef:
    cik: str
    accession: str
    form: str
    accepted_at: datetime
    period_end: str


class SecEdgarProvider(Protocol):
    def filings(self, cik: str, start: str, end: str) -> list[FilingRef]: ...

    def mda_locator(self, accession: str) -> dict[str, str]: ...


class PointInTimeMarketProvider(Protocol):
    def financials(self, security_id: str, available_at: datetime) -> dict[str, float]: ...

    def market_state(self, security_id: str, available_at: datetime) -> dict[str, float | str]: ...

    def outcomes(self, security_id: str, entry_date: str, exit_date: str) -> dict[str, float | bool]: ...


class UnconfiguredLicensedProvider:
    """Safe default that prevents accidental scraping or unlicensed data use."""

    def _unavailable(self) -> None:
        raise DataAccessUnavailable(
            "Configure a licensed CRSP/Compustat-compatible provider outside this repository."
        )

    def financials(self, security_id: str, available_at: datetime) -> dict[str, float]:
        self._unavailable()

    def market_state(self, security_id: str, available_at: datetime) -> dict[str, float | str]:
        self._unavailable()

    def outcomes(self, security_id: str, entry_date: str, exit_date: str) -> dict[str, float | bool]:
        self._unavailable()
