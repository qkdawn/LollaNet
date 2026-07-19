"""Leakage-audited loader and train-only preprocessor for router experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


ROUTE_RULES = ("evidence_revision", "balance_sheet_margin", "operating_constraint", "fragile_estimate")
FINANCIAL_FIELDS = (
    "ttm_revenue_growth", "gross_margin_change", "ttm_operating_margin", "ttm_operating_cash_flow",
    "ttm_capex", "ttm_free_cash_flow", "free_cash_flow_positive_years", "roic", "net_debt_to_ebitda",
    "interest_coverage", "current_ratio", "receivables_growth_minus_revenue_growth",
    "inventory_growth_minus_revenue_growth", "capex_to_revenue_change", "earnings_surprise_zscore",
    "earnings_volatility", "operating_cash_flow_sign_changed",
)
MARKET_FIELDS = ("market_cap", "free_cash_flow_yield", "valuation_percentile", "trailing_drawdown")
NUMERIC_FIELDS = FINANCIAL_FIELDS + MARKET_FIELDS
MONEY_FIELDS = {"ttm_operating_cash_flow", "ttm_capex", "ttm_free_cash_flow", "market_cap"}
FORBIDDEN_INPUT_KEYS = {"cik", "issuer_name", "ticker_at_collection", "exploratory_outcome", "outcome", "filing"}
REQUIRED_PROVENANCE_KEYS = {"cik", "issuer_name", "ticker_at_collection", "exploratory_outcome", "filing"}


def split_for(available_on: str) -> str | None:
    year = int(available_on[:4])
    if 2017 <= year <= 2020:
        return "train"
    if year == 2021:
        return "development"
    if 2022 <= year <= 2023:
        return "sealed_test"
    return None


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records = []
    for line in raw.splitlines():
        if not line:
            continue
        record = json.loads(line)
        split = split_for(record["availability"]["public_available_at"])
        if split is None:
            continue
        if REQUIRED_PROVENANCE_KEYS - set(record):
            raise ValueError("candidate record is missing expected provenance fields")
        record["_split"] = split
        records.append(record)
    if not records:
        raise ValueError("no records fall within the 2017-2023 router split")
    return records, sha256(raw).hexdigest()


def rule_is_observed(record: dict[str, Any], rule_id: str) -> bool:
    return all(item["rule_id"] != rule_id for item in record["route_labels"].get("unavailable_rules", []))


def _value(record: dict[str, Any], field: str) -> float | None:
    value = (record["financial_features"] | record["market_state"]).get(field)
    if value is None:
        return None
    return float(value)


@dataclass
class RouterPreprocessor:
    low: np.ndarray
    high: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sectors: list[str]

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "RouterPreprocessor":
        rows = np.array([[_value(record, field) if _value(record, field) is not None else np.nan for field in NUMERIC_FIELDS] for record in records], dtype=np.float64)
        transformed = _transform_money(rows)
        low = np.zeros(len(NUMERIC_FIELDS), dtype=np.float64)
        high = np.zeros(len(NUMERIC_FIELDS), dtype=np.float64)
        median = np.zeros(len(NUMERIC_FIELDS), dtype=np.float64)
        mean = np.zeros(len(NUMERIC_FIELDS), dtype=np.float64)
        std = np.ones(len(NUMERIC_FIELDS), dtype=np.float64)
        for index in range(len(NUMERIC_FIELDS)):
            observed = transformed[:, index][~np.isnan(transformed[:, index])]
            if not len(observed):
                continue
            low[index], high[index] = np.quantile(observed, [0.01, 0.99])
            clipped = np.clip(observed, low[index], high[index])
            median[index] = np.median(clipped)
            mean[index] = clipped.mean()
            std[index] = clipped.std() or 1.0
        clipped = np.clip(transformed, low, high)
        filled = np.where(np.isnan(clipped), median, clipped)
        sectors = sorted({str(record["market_state"].get("sector_code") or "unknown") for record in records})
        return cls(low, high, median, mean, std, sectors)

    def transform(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.array([[_value(record, field) if _value(record, field) is not None else np.nan for field in NUMERIC_FIELDS] for record in records], dtype=np.float64)
        missing = np.isnan(values).astype(np.float32)
        values = _transform_money(values)
        values = np.clip(values, self.low, self.high)
        values = np.where(np.isnan(values), self.median, values)
        standardized = ((values - self.mean) / self.std).astype(np.float32)
        sector_index = {name: index for index, name in enumerate(self.sectors)}
        sectors = np.array([sector_index.get(str(record["market_state"].get("sector_code") or "unknown"), sector_index.get("unknown", 0)) for record in records], dtype=np.int64)
        return np.concatenate((standardized, missing), axis=1), sectors, missing

    def to_dict(self) -> dict[str, Any]:
        return {"numeric_fields": list(NUMERIC_FIELDS), "money_fields": sorted(MONEY_FIELDS), "low": self.low.tolist(), "high": self.high.tolist(), "median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist(), "sectors": self.sectors}


def _transform_money(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, field in enumerate(NUMERIC_FIELDS):
        if field in MONEY_FIELDS:
            column = result[:, index]
            result[:, index] = np.sign(column) * np.log1p(np.abs(column))
    return result


def targets(records: list[dict[str, Any]], return_clip: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, tuple[float, float] | None]:
    rule_y = np.zeros((len(records), len(ROUTE_RULES)), dtype=np.float32)
    rule_mask = np.zeros_like(rule_y)
    returns = np.zeros(len(records), dtype=np.float32)
    return_mask = np.zeros(len(records), dtype=np.float32)
    raw_returns = []
    for index, record in enumerate(records):
        triggered = set(record["route_labels"].get("triggered_rules", []))
        for rule_index, rule in enumerate(ROUTE_RULES):
            if rule_is_observed(record, rule):
                rule_mask[index, rule_index] = 1.0
                rule_y[index, rule_index] = float(rule in triggered)
        outcome = record.get("exploratory_outcome", {})
        if outcome.get("status") == "exploratory_price_only" and outcome.get("raw_price_return") is not None and float(outcome["raw_price_return"]) > -1:
            raw_returns.append(float(np.log1p(float(outcome["raw_price_return"]))))
    if raw_returns and return_clip is None:
        return_clip = tuple(np.quantile(raw_returns, [0.01, 0.99]).tolist())
    if raw_returns:
        clipped_train = np.clip(raw_returns, return_clip[0], return_clip[1])  # type: ignore[index]
        return_mean_std = (float(np.mean(clipped_train)), float(np.std(clipped_train) or 1.0))
    else:
        return_mean_std = None
    for index, record in enumerate(records):
        outcome = record.get("exploratory_outcome", {})
        if outcome.get("status") == "exploratory_price_only" and outcome.get("raw_price_return") is not None and float(outcome["raw_price_return"]) > -1 and return_clip and return_mean_std:
            value = float(np.log1p(float(outcome["raw_price_return"])))
            returns[index] = (float(np.clip(value, return_clip[0], return_clip[1])) - return_mean_std[0]) / return_mean_std[1]
            return_mask[index] = 1.0
    return rule_y, rule_mask, returns, return_mask, return_clip, return_mean_std


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}, "positive_labels": {}, "rule_observed": {}}
    for split in ("train", "development", "sealed_test"):
        subset = [record for record in records if record["_split"] == split]
        result["splits"][split] = len(subset)
        for rule in ROUTE_RULES:
            result["positive_labels"].setdefault(rule, {})[split] = sum(rule in record["route_labels"].get("triggered_rules", []) for record in subset)
            result["rule_observed"].setdefault(rule, {})[split] = sum(rule_is_observed(record, rule) for record in subset)
    for rule in ROUTE_RULES:
        if result["positive_labels"][rule]["train"] < 50 or result["positive_labels"][rule]["development"] < 20 or result["positive_labels"][rule]["sealed_test"] < 20:
            raise ValueError(f"{rule} does not meet the minimum positive-label threshold")
    return result
