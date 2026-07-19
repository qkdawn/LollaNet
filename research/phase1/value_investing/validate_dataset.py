"""Validate local value-investing fixtures before any licensed data is ingested."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Any

import yaml

try:
    from .route_labels import derive_route, load_rules
except ImportError:  # Supports direct execution from the repository root.
    from route_labels import derive_route, load_rules


ROOT = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _split_for(available_on: date, splits: dict[str, list[Any]]) -> str | None:
    for split, (start, end) in splits.items():
        start_date = _date(str(start))
        end_date = _date(str(end))
        if start_date <= available_on <= end_date:
            return split
    return None


def validate_episode(episode: dict[str, Any], contract: dict[str, Any], feature_dictionary: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = set(contract["required_episode_fields"]) - set(episode)
    if missing:
        errors.append(f"missing episode fields {sorted(missing)}")
    forbidden = set(contract["forbidden_episode_fields"]) & set(episode)
    if forbidden:
        errors.append(f"contains forbidden episode fields {sorted(forbidden)}")
    if missing:
        return errors

    security = episode["security"]
    if security.get("security_type") != "common_stock" or security.get("is_financial") is not False:
        errors.append("episode must represent a non-financial common share")

    availability = episode["availability"]
    try:
        accepted = _date(availability["filing_accepted_at"])
        available = _date(availability["public_available_at"])
        entry = _date(availability["entry_date"])
        if accepted > available:
            errors.append("filing acceptance cannot follow public availability")
        if entry <= available:
            errors.append("entry date must be after public availability")
        expected_split = _split_for(available, contract["splits"])
        if episode["split"] != expected_split:
            errors.append(f"split {episode['split']} does not match availability timestamp")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid availability fields: {exc}")
        entry = None

    sources = episode["sources"]
    for source_name in ("filing", "financials", "market"):
        source = sources.get(source_name)
        if not isinstance(source, dict) or not source.get("provider") or not source.get("record_version"):
            errors.append(f"missing provenance for {source_name}")
        elif source_name != "filing" and not source.get("source_fields"):
            errors.append(f"missing source fields for {source_name}")
    locator = episode["mda_locator"]
    required_locator = {"accession", "section", "start_char", "end_char", "text_sha256", "extractor_version"}
    if not required_locator <= set(locator) or "text" in locator:
        errors.append("MD&A locator must be a text-free, versioned pointer")

    expected_financial = {field["name"] for field in feature_dictionary["financial_features"]}
    expected_market = {field["name"] for field in feature_dictionary["market_state"]}
    if set(episode["financial_features"]) != expected_financial:
        errors.append("financial feature set does not match the feature dictionary")
    if set(episode["market_state"]) != expected_market:
        errors.append("market feature set does not match the feature dictionary")

    derived = derive_route(episode["financial_features"] | episode["market_state"], rules)
    if episode["route_labels"] != derived:
        errors.append("route labels do not match the pre-registered rule output")
    activated = set(episode["route_labels"].get("activated_models", []))
    if not activated <= set(rules["allowed_model_ids"]):
        errors.append("route activates a model outside the financial-statement-observable subgraph")

    outcome = episode["outcome"]
    required_outcome = {
        "exit_date", "entry_close", "exit_close", "total_return", "market_return", "market_excess_return",
        "sector_relative_return", "maximum_drawdown", "tail_loss", "delisting_return", "delisting_return_included",
    }
    if not required_outcome <= set(outcome):
        errors.append("outcome is incomplete")
    elif entry is not None:
        try:
            if (_date(outcome["exit_date"]) - entry).days < 365:
                errors.append("outcome does not cover the 12-month holding window")
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid outcome date: {exc}")
    if outcome.get("delisting_return_included") is not True:
        errors.append("outcome must explicitly include delisting return treatment")
    return errors


def validate_fixture(root: Path = ROOT) -> list[str]:
    contract = _load_yaml(root / "data_contract.yaml")
    dictionary = _load_yaml(root / "feature_dictionary.yaml")
    rules = load_rules(str(root / "route_rules.yaml"))
    episode = json.loads((root / "fixtures" / "anonymous_episode.json").read_text(encoding="utf-8"))
    return validate_episode(episode, contract, dictionary, rules)


def main() -> None:
    errors = validate_fixture()
    if errors:
        print("Value-investing fixture validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("Value-investing fixture is valid.")


if __name__ == "__main__":
    main()
