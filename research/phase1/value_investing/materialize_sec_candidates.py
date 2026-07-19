"""Materialize point-in-time SEC financial-statement route-training candidates.

The output is deliberately *not* a completed investment Episode: SEC Company
Facts alone cannot establish common-share eligibility, market state, MD&A text,
or delisting-adjusted outcomes.  Every unavailable field is explicit so later
providers can augment candidates without rewriting their financial provenance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

try:
    from .download_free_pilot import external_output
    from .route_labels import derive_route_with_coverage, load_rules
except ImportError:  # Supports direct execution from the repository root.
    from download_free_pilot import external_output
    from route_labels import derive_route_with_coverage, load_rules


ROOT = Path(__file__).resolve().parent
FLOW_TAGS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "interest_expense": ("InterestExpenseNonoperating", "InterestExpense"),
    "net_income": ("NetIncomeLoss",),
    "depreciation": ("DepreciationDepletionAndAmortization", "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"),
}
INSTANT_TAGS = {
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "debt_current": ("ShortTermBorrowings", "LongTermDebtCurrent"),
    "debt_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "receivables": ("AccountsReceivableNetCurrent",),
    "inventory": ("InventoryNet", "InventoryFinishedGoodsNetOfReserves"),
}
FEATURES = (
    "ttm_revenue_growth", "gross_margin_change", "ttm_operating_margin", "ttm_operating_cash_flow",
    "ttm_capex", "ttm_free_cash_flow", "free_cash_flow_positive_years", "roic", "net_debt_to_ebitda",
    "interest_coverage", "current_ratio", "receivables_growth_minus_revenue_growth",
    "inventory_growth_minus_revenue_growth", "capex_to_revenue_change", "earnings_surprise_zscore",
    "earnings_volatility", "operating_cash_flow_sign_changed",
)
MARKET_FEATURES = ("market_cap", "free_cash_flow_yield", "valuation_percentile", "sector_code", "trailing_drawdown")


def _day(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _allowed(row: dict[str, Any], filing_date: date, accession: str) -> bool:
    if row.get("form") not in {"10-K", "10-Q"} or not row.get("filed"):
        return False
    return row.get("accn") == accession or _day(row["filed"]) < filing_date


def _rows(facts: dict[str, Any], tags: Iterable[str], filing_date: date, accession: str) -> list[dict[str, Any]]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    collected = []
    for tag in tags:
        for unit, values in gaap.get(tag, {}).get("units", {}).items():
            if unit not in {"USD", "USD/shares"}:
                continue
            for value in values:
                if "val" in value and value.get("end") and _allowed(value, filing_date, accession):
                    collected.append({**value, "tag": tag, "unit": unit})
    return collected


def _best_instant(facts: dict[str, Any], tags: Iterable[str], filing_date: date, accession: str) -> tuple[float | None, dict[str, Any] | None]:
    rows = [row for row in _rows(facts, tags, filing_date, accession) if not row.get("start")]
    if not rows:
        return None, None
    best = max(rows, key=lambda row: (row["end"], row.get("filed", ""), row.get("accn") == accession))
    return float(best["val"]), best


def _quarterly_values(facts: dict[str, Any], tags: Iterable[str], filing_date: date, accession: str) -> list[tuple[date, float, dict[str, Any]]]:
    """Build quarterly flows, deriving Q4 only from an annual value and three known quarters."""
    rows = [row for row in _rows(facts, tags, filing_date, accession) if row.get("start")]
    direct: dict[date, tuple[float, dict[str, Any]]] = {}
    annual = []
    for row in rows:
        duration = (_day(row["end"]) - _day(row["start"])).days + 1
        end = _day(row["end"])
        if 70 <= duration <= 115:
            old = direct.get(end)
            if old is None or (row.get("filed", ""), row.get("accn") == accession) > (old[1].get("filed", ""), old[1].get("accn") == accession):
                direct[end] = (float(row["val"]), row)
        elif 300 <= duration <= 390 and row.get("form") == "10-K":
            annual.append((end, float(row["val"]), row))
    result = dict(direct)
    for end, value, row in annual:
        prior = [(day, amount) for day, (amount, _source) in direct.items() if end - timedelta(days=310) < day < end]
        if len(prior) >= 3:
            q4 = value - sum(amount for _day_value, amount in sorted(prior)[-3:])
            result.setdefault(end, (q4, {**row, "derived_q4": True}))
    return [(end, amount, source) for end, (amount, source) in sorted(result.items())]


def _ttm(series: list[tuple[date, float, dict[str, Any]]]) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    if len(series) < 8:
        return None, None, []
    current = series[-4:]
    prior = series[-8:-4]
    return sum(item[1] for item in current), sum(item[1] for item in prior), [item[2] for item in current]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _feature(value: float | bool | None, reason: str | None = None) -> tuple[float | bool | None, str | None]:
    return value, reason if value is None else None


def financial_snapshot(facts: dict[str, Any], filing_date: date, accession: str) -> tuple[dict[str, Any], dict[str, Any]]:
    flows = {name: _quarterly_values(facts, tags, filing_date, accession) for name, tags in FLOW_TAGS.items()}
    ttm = {name: _ttm(values) for name, values in flows.items()}
    revenue, prior_revenue, revenue_sources = ttm["revenue"]
    operating_income, _prior_operating_income, operating_sources = ttm["operating_income"]
    cash_flow, prior_cash_flow, cash_flow_sources = ttm["operating_cash_flow"]
    capex, prior_capex, capex_sources = ttm["capex"]
    cost, prior_cost, cost_sources = ttm["cost_of_revenue"]
    interest, _prior_interest, interest_sources = ttm["interest_expense"]
    depreciation, _prior_depreciation, depreciation_sources = ttm["depreciation"]
    net_income_series = flows["net_income"]
    cash, cash_source = _best_instant(facts, INSTANT_TAGS["cash"], filing_date, accession)
    debt_current, debt_current_source = _best_instant(facts, INSTANT_TAGS["debt_current"], filing_date, accession)
    debt_noncurrent, debt_noncurrent_source = _best_instant(facts, INSTANT_TAGS["debt_noncurrent"], filing_date, accession)
    current_assets, current_assets_source = _best_instant(facts, INSTANT_TAGS["current_assets"], filing_date, accession)
    current_liabilities, current_liabilities_source = _best_instant(facts, INSTANT_TAGS["current_liabilities"], filing_date, accession)
    receivables, receivables_source = _best_instant(facts, INSTANT_TAGS["receivables"], filing_date, accession)
    inventory, inventory_source = _best_instant(facts, INSTANT_TAGS["inventory"], filing_date, accession)
    # The previous balance snapshot is deliberately selected before the current period end.
    period_end = max((source.get("end") for source in [receivables_source, inventory_source, current_assets_source] if source), default=None)
    previous_date = _day(period_end) - timedelta(days=80) if period_end else filing_date
    def previous(tags: Iterable[str]) -> float | None:
        rows = [row for row in _rows(facts, tags, filing_date, accession) if not row.get("start") and _day(row["end"]) <= previous_date]
        return float(max(rows, key=lambda row: row["end"])["val"]) if rows else None
    previous_receivables, previous_inventory = previous(INSTANT_TAGS["receivables"]), previous(INSTANT_TAGS["inventory"])
    fcf = cash_flow - abs(capex) if cash_flow is not None and capex is not None else None
    previous_fcf = prior_cash_flow - abs(prior_capex) if prior_cash_flow is not None and prior_capex is not None else None
    current_margin = _ratio(revenue - cost if revenue is not None and cost is not None else None, revenue)
    previous_margin = _ratio(prior_revenue - prior_cost if prior_revenue is not None and prior_cost is not None else None, prior_revenue)
    debt = (debt_current or 0) + (debt_noncurrent or 0) if debt_current is not None or debt_noncurrent is not None else None
    ebitda = operating_income + depreciation if operating_income is not None and depreciation is not None else None
    quarterly_income = [value for _end, value, _source in net_income_series[-12:]]
    surprise = None
    volatility = None
    if len(quarterly_income) >= 8:
        history = quarterly_income[-8:-1]
        spread = statistics.pstdev(history)
        surprise = (quarterly_income[-1] - statistics.mean(history)) / spread if spread else None
        scale = abs(statistics.mean(quarterly_income))
        volatility = statistics.pstdev(quarterly_income) / scale if scale else None
    current_capex_ratio = _ratio(abs(capex) if capex is not None else None, revenue)
    prior_capex_ratio = _ratio(abs(prior_capex) if prior_capex is not None else None, prior_revenue)
    features = {
        "ttm_revenue_growth": _ratio(revenue - prior_revenue if revenue is not None and prior_revenue is not None else None, prior_revenue),
        "gross_margin_change": current_margin - previous_margin if current_margin is not None and previous_margin is not None else None,
        "ttm_operating_margin": _ratio(operating_income, revenue),
        "ttm_operating_cash_flow": cash_flow,
        "ttm_capex": abs(capex) if capex is not None else None,
        "ttm_free_cash_flow": fcf,
        "free_cash_flow_positive_years": None,  # Requires three annual fiscal-period reconstructions.
        "roic": None,  # Requires tax and invested-capital normalization beyond a universal XBRL tag map.
        "net_debt_to_ebitda": _ratio(debt - cash if debt is not None and cash is not None else None, ebitda),
        "interest_coverage": _ratio(operating_income, abs(interest) if interest is not None else None),
        "current_ratio": _ratio(current_assets, current_liabilities),
        "receivables_growth_minus_revenue_growth": _ratio(receivables - previous_receivables if receivables is not None and previous_receivables is not None else None, previous_receivables) - _ratio(revenue - prior_revenue if revenue is not None and prior_revenue is not None else None, prior_revenue) if receivables is not None and previous_receivables not in (None, 0) and revenue is not None and prior_revenue not in (None, 0) else None,
        "inventory_growth_minus_revenue_growth": _ratio(inventory - previous_inventory if inventory is not None and previous_inventory is not None else None, previous_inventory) - _ratio(revenue - prior_revenue if revenue is not None and prior_revenue is not None else None, prior_revenue) if inventory is not None and previous_inventory not in (None, 0) and revenue is not None and prior_revenue not in (None, 0) else None,
        "capex_to_revenue_change": current_capex_ratio - prior_capex_ratio if current_capex_ratio is not None and prior_capex_ratio is not None else None,
        "earnings_surprise_zscore": surprise,
        "earnings_volatility": volatility,
        "operating_cash_flow_sign_changed": cash_flow * prior_cash_flow < 0 if cash_flow is not None and prior_cash_flow is not None else None,
    }
    source_rows = [*revenue_sources, *operating_sources, *cash_flow_sources, *capex_sources, *cost_sources, *interest_sources, *depreciation_sources, cash_source, debt_current_source, debt_noncurrent_source, current_assets_source, current_liabilities_source, receivables_source, inventory_source]
    provenance = {"provider": "SEC EDGAR Company Facts", "record_version": "companyfacts-v1", "source_fields": sorted({row["tag"] for row in source_rows if row}), "facts_filtered_before_filing_date": True}
    missing = {name: "not derivable from the available SEC XBRL observations" for name, value in features.items() if value is None}
    return features, {"financials": provenance, "missing_reasons": missing}


def filings(submissions: dict[str, Any], start: int, end: int) -> Iterable[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    for index, form in enumerate(recent.get("form", [])):
        if form not in {"10-K", "10-Q"}:
            continue
        accepted = recent.get("acceptanceDateTime", [None])[index]
        accession = recent.get("accessionNumber", [None])[index]
        if not accepted or not accession or not (start <= _day(accepted).year <= end):
            continue
        yield {"form": form, "accepted": accepted, "accession": accession, "period_end": recent.get("reportDate", [None])[index]}


def materialize(raw_root: Path, output: Path, start: int = 2005, end: int = 2023) -> dict[str, Any]:
    raw_root, output = external_output(raw_root), external_output(output)
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    rules = load_rules(str(ROOT / "route_rules.yaml"))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "sec_financial_candidates.jsonl"
    coverage: dict[str, int] = defaultdict(int)
    total = 0
    with destination.open("w", encoding="utf-8") as handle:
        for cik, issuer in manifest["issuers"].items():
            if issuer.get("status") != "accepted":
                continue
            issuer_dir = raw_root / "issuers" / cik
            facts = json.loads((issuer_dir / "company_facts.json").read_text(encoding="utf-8"))
            submissions = json.loads((issuer_dir / "submissions.json").read_text(encoding="utf-8"))
            for filing in filings(submissions, start, end):
                filing_date = _day(filing["accepted"])
                features, evidence = financial_snapshot(facts, filing_date, filing["accession"])
                market = {name: None for name in MARKET_FEATURES}
                labels = derive_route_with_coverage(features | market, rules)
                available = {name: value for name, value in (features | market).items() if value is not None}
                for name in available:
                    coverage[name] += 1
                episode_id = hashlib.sha256(f"{cik}:{filing['accession']}".encode()).hexdigest()[:20]
                record = {
                    "record_type": "sec_financial_candidate_v1",
                    "episode_id": f"sec-{episode_id}",
                    "cik": cik,
                    "ticker_at_collection": issuer["ticker"],
                    "issuer_name": issuer["title"],
                    "sic": issuer.get("sic"),
                    "security_type": "unverified_public_equity",
                    "is_financial": False,
                    "filing": filing,
                    "availability": {"filing_accepted_at": filing["accepted"], "public_available_at": filing["accepted"]},
                    "financial_features": features,
                    "market_state": market,
                    "provenance": evidence["financials"],
                    "missing_reasons": {**evidence["missing_reasons"], **{name: "requires an external point-in-time market-data provider" for name in MARKET_FEATURES}},
                    "mda_locator": {"status": "not_collected", "accession": filing["accession"]},
                    "route_labels": labels,
                    "training_status": "provisional_missing_market_and_outcome",
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                total += 1
    report = {"record_type": "sec_candidate_coverage_v1", "raw_root": str(raw_root), "candidate_count": total, "feature_available_count": dict(sorted(coverage.items())), "market_features_missing_for_all": list(MARKET_FEATURES), "required_next_sources": ["point-in-time price and common-share mapping", "delisting-adjusted return provider", "MD&A section extractor"]}
    (output / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    report = materialize(args.raw_root, args.output, args.start_year, args.end_year)
    print(f"Materialized {report['candidate_count']} SEC financial candidates to {external_output(args.output)}")


if __name__ == "__main__":
    main()
