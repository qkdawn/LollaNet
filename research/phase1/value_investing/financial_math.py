"""Small deterministic calculations used by point-in-time data adapters."""

from __future__ import annotations

from collections.abc import Sequence


def trailing_twelve_months(quarter_values: Sequence[float]) -> float:
    if len(quarter_values) != 4:
        raise ValueError("TTM requires exactly four reported fiscal quarters")
    return float(sum(quarter_values))


def delisting_adjusted_total_return(entry_close: float, exit_close: float, delisting_return: float) -> float:
    if entry_close <= 0 or exit_close < 0:
        raise ValueError("prices must be non-negative and entry price must be positive")
    return (exit_close / entry_close) * (1.0 + delisting_return) - 1.0


def maximum_drawdown(closes: Sequence[float]) -> float:
    if not closes or any(price <= 0 for price in closes):
        raise ValueError("drawdown requires positive close prices")
    peak = closes[0]
    drawdown = 0.0
    for price in closes:
        peak = max(peak, price)
        drawdown = min(drawdown, price / peak - 1.0)
    return drawdown
