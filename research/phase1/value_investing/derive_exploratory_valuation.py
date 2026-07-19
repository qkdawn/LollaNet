"""Derive no-lookahead SIC-peer valuation percentiles for free-price candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

try:
    from .download_free_pilot import external_output
    from .route_labels import derive_route_with_coverage, load_rules
except ImportError:
    from download_free_pilot import external_output
    from route_labels import derive_route_with_coverage, load_rules


ROOT = Path(__file__).resolve().parent


def derive(input_path: Path, output: Path, minimum_peers: int = 10) -> dict[str, Any]:
    input_path, output = external_output(input_path), external_output(output)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    rules = load_rules(str(ROOT / "route_rules.yaml"))
    history: dict[str, list[float]] = defaultdict(list)
    assigned = 0
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_day[record["availability"]["public_available_at"][:10]].append(record)
    for available_on in sorted(by_day):
        group = by_day[available_on]
        for record in group:
            market = record["market_state"]
            sector, value = market.get("sector_code"), market.get("free_cash_flow_yield")
            peers = history[sector] if sector else []
            if value is not None and len(peers) >= minimum_peers:
                # Higher FCF yield means cheaper on this explicitly declared proxy.
                market["valuation_percentile"] = sum(peer <= value for peer in peers) / len(peers)
                assigned += 1
            else:
                market["valuation_percentile"] = None
            record["route_labels"] = derive_route_with_coverage(record["financial_features"] | market, rules)
        for record in group:
            market = record["market_state"]
            if market.get("sector_code") and market.get("free_cash_flow_yield") is not None:
                history[market["sector_code"]].append(market["free_cash_flow_yield"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "sec_candidates_market_valued.jsonl"
    destination.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    report = {"record_type": "exploratory_valuation_v1", "candidate_count": len(records), "valuation_percentile_available": assigned, "minimum_prior_sic_peers": minimum_peers, "no_lookahead": "same-day records receive no access to each other; only earlier public availability dates enter the peer history", "limitations": ["free prices use unverified current ticker mapping", "SIC is a coarse sector proxy", "not suitable for research-grade investment-performance claims"]}
    (output / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-peers", type=int, default=10)
    args = parser.parse_args()
    report = derive(args.input, args.output, args.minimum_peers)
    print(f"Derived no-lookahead valuation percentiles for {report['valuation_percentile_available']} candidates")


if __name__ == "__main__":
    main()
