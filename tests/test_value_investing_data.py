import json

import pytest
import yaml

from research.phase1.value_investing.adapters import DataAccessUnavailable, UnconfiguredLicensedProvider
from research.phase1.value_investing.financial_math import (
    delisting_adjusted_total_return,
    maximum_drawdown,
    trailing_twelve_months,
)
from research.phase1.value_investing.free_sources import (
    AlphaVantageFreePriceClient,
    FreeSourceConfigurationError,
    SecEdgarFreeClient,
    StooqFreePriceClient,
    YahooFinanceFreePriceClient,
    facts_available_as_of,
)
from research.phase1.value_investing.download_free_pilot import external_output
from research.phase1.value_investing.collect_sec_universe import (
    Candidate,
    RequestPacer,
    collect,
    is_nonfinancial,
    select_candidates,
)
from research.phase1.value_investing.route_labels import derive_route, load_rules
from research.phase1.value_investing.router_data import ROUTE_RULES, RouterPreprocessor, audit_records, rule_is_observed, split_for, targets
from research.phase1.value_investing.router_decode import decode_route
from research.phase1.value_investing.prepare_llm_teacher_sample import FORBIDDEN_PACKET_KEYS, _packet
from research.phase1.value_investing.validate_llm_teacher_labels import validate_label
from research.phase1.value_investing.validate_dataset import ROOT, validate_episode, validate_fixture


def load_fixture():
    return json.loads((ROOT / "fixtures" / "anonymous_episode.json").read_text(encoding="utf-8"))


def load_contract_material():
    contract = yaml.safe_load((ROOT / "data_contract.yaml").read_text(encoding="utf-8"))
    dictionary = yaml.safe_load((ROOT / "feature_dictionary.yaml").read_text(encoding="utf-8"))
    rules = load_rules(str(ROOT / "route_rules.yaml"))
    return contract, dictionary, rules


def test_value_investing_fixture_is_point_in_time_valid():
    assert validate_fixture() == []


def test_validator_rejects_entry_before_public_availability():
    episode = load_fixture()
    contract, dictionary, rules = load_contract_material()
    episode["availability"]["entry_date"] = "2015-05-14"
    assert "entry date must be after public availability" in validate_episode(episode, contract, dictionary, rules)


def test_validator_rejects_route_label_tampering_and_raw_mda_text():
    episode = load_fixture()
    contract, dictionary, rules = load_contract_material()
    episode["route_labels"]["activated_models"] = []
    episode["raw_mda_text"] = "forbidden"
    errors = validate_episode(episode, contract, dictionary, rules)
    assert "route labels do not match the pre-registered rule output" in errors
    assert any("forbidden episode fields" in error for error in errors)


def test_route_rules_do_not_activate_physics_or_chemistry_models():
    episode = load_fixture()
    _, _, rules = load_contract_material()
    labels = derive_route(episode["financial_features"] | episode["market_state"], rules)
    assert set(labels["activated_models"]) <= set(rules["allowed_model_ids"])
    assert "mm-p1-force-balance" not in labels["activated_models"]
    assert labels["ordered_models"].index("mm-p1-expected-value") < labels["ordered_models"].index("mm-p1-opportunity-cost")


def test_financial_math_requires_complete_ttm_and_includes_delisting_return():
    assert trailing_twelve_months([1.0, 2.0, 3.0, 4.0]) == 10.0
    with pytest.raises(ValueError):
        trailing_twelve_months([1.0, 2.0, 3.0])
    assert delisting_adjusted_total_return(100.0, 110.0, -0.5) == pytest.approx(-0.45)
    assert maximum_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(-0.25)


def test_unconfigured_provider_cannot_fetch_licensed_data():
    provider = UnconfiguredLicensedProvider()
    with pytest.raises(DataAccessUnavailable):
        provider.market_state("fixture", None)  # type: ignore[arg-type]


def test_sec_free_client_requires_contact_and_uses_configured_agent():
    with pytest.raises(FreeSourceConfigurationError):
        SecEdgarFreeClient(user_agent="prototype")

    seen = []

    def fetch(request):
        seen.append(request.headers["User-agent"])
        return b'{"0": {"ticker": "TEST"}}'

    client = SecEdgarFreeClient("LollaNet Research test@example.edu", fetch=fetch)
    assert client.company_tickers()["0"]["ticker"] == "TEST"
    assert seen == ["LollaNet Research test@example.edu"]


def test_sec_facts_as_of_excludes_later_filing_revisions():
    company_facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "filed": "2020-05-01", "end": "2020-03-31", "val": 100, "accn": "old"},
                            {"form": "10-Q", "filed": "2021-05-01", "end": "2020-03-31", "val": 120, "accn": "restated"},
                        ]
                    }
                }
            }
        }
    }
    facts = facts_available_as_of(company_facts, ["Revenues"], "2020-06-01")
    assert facts["Revenues"].value == 100.0
    assert facts["Revenues"].accession == "old"


def test_stooq_client_parses_only_valid_daily_closes():
    def fetch(_request):
        return b"Date,Open,High,Low,Close,Volume\n2020-01-02,10,11,9,10.5,100\n2020-01-03,N/D,N/D,N/D,N/D,0\n"

    closes = StooqFreePriceClient(fetch=fetch).daily_closes("TEST")
    assert [(item.trading_date, item.close) for item in closes] == [("2020-01-02", 10.5)]


def test_stooq_client_rejects_an_anti_automation_page():
    with pytest.raises(FreeSourceConfigurationError, match="anti-automation"):
        StooqFreePriceClient(fetch=lambda _request: b"<html>blocked</html>").daily_closes("TEST")


def test_alpha_vantage_client_requires_key_and_parses_monthly_closes():
    with pytest.raises(FreeSourceConfigurationError, match="ALPHAVANTAGE_API_KEY"):
        AlphaVantageFreePriceClient(api_key="")

    def fetch(_request):
        return b'{"Monthly Time Series":{"2020-01-02":{"4. close":"10.50"}}}'

    closes = AlphaVantageFreePriceClient("fixture-key", fetch=fetch).monthly_closes("test")
    assert [(item.trading_date, item.close) for item in closes] == [("2020-01-02", 10.5)]


def test_yahoo_client_parses_adjusted_daily_closes():
    def fetch(_request):
        return b'{"chart":{"result":[{"timestamp":[1577923200],"indicators":{"adjclose":[{"adjclose":[10.5]}]}}],"error":null}}'

    closes = YahooFinanceFreePriceClient(fetch=fetch).daily_adjusted_closes("test", 0, 1)
    assert [(item.trading_date, item.close) for item in closes] == [("2020-01-02", 10.5)]


def test_alpha_vantage_client_parses_monthly_adjusted_closes():
    def fetch(_request):
        return b'{"Monthly Adjusted Time Series":{"2020-01-02":{"4. close":"10.50","5. adjusted close":"10.25"}}}'

    closes = AlphaVantageFreePriceClient("fixture-key", fetch=fetch).monthly_adjusted_closes("test")
    assert [(item.trading_date, item.close, item.adjusted_close) for item in closes] == [("2020-01-02", 10.5, 10.25)]


def test_free_pilot_refuses_to_write_downloads_inside_repository(tmp_path):
    with pytest.raises(ValueError):
        external_output(ROOT / "fixtures" / "downloaded")
    assert external_output(tmp_path) == tmp_path.resolve()


def test_sec_universe_selection_is_seeded_and_deduplicates_ciks():
    tickers = {
        "0": {"cik_str": 1, "ticker": "ONE", "title": "One"},
        "1": {"cik_str": "0000000001", "ticker": "DUP", "title": "Duplicate"},
        "2": {"cik_str": 2, "ticker": "TWO", "title": "Two"},
    }
    first = select_candidates(tickers, target=2, seed=7)
    assert first == select_candidates(tickers, target=2, seed=7)
    assert {item.cik for item in first} == {"0000000001", "0000000002"}


def test_sec_universe_excludes_financial_sics_and_short_filing_histories():
    assert is_nonfinancial({"sic": "6021", "filings": {"recent": {"form": ["10-K", "10-Q"]}}}) == (False, "financial_sic")
    assert is_nonfinancial({"sic": "3570", "filings": {"recent": {"form": ["10-K"]}}}) == (False, "insufficient_10k_10q_history")
    assert is_nonfinancial({"sic": "3570", "filings": {"recent": {"form": ["10-K", "10-Q"]}}}) == (True, "accepted")


def test_sec_universe_collector_writes_external_raw_data_and_resumes(tmp_path):
    class FakeSec:
        def __init__(self):
            self.submission_calls = 0
            self.fact_calls = 0

        def company_tickers(self):
            return {"0": {"cik_str": 1, "ticker": "ONE", "title": "One Inc"}}

        def submissions(self, _cik):
            self.submission_calls += 1
            return {"sic": "3570", "sicDescription": "Computers", "filings": {"recent": {"form": ["10-K", "10-Q"]}}}

        def company_facts(self, _cik):
            self.fact_calls += 1
            return {"facts": {}}

    client = FakeSec()
    manifest = collect(1, tmp_path, seed=1, client=client)
    item = manifest["issuers"]["0000000001"]
    assert item["status"] == "accepted"
    assert (tmp_path / item["paths"]["submissions"]).exists()
    assert (tmp_path / item["paths"]["company_facts"]).exists()
    collect(1, tmp_path, seed=1, client=client)
    assert (client.submission_calls, client.fact_calls) == (1, 1)


def test_request_pacer_sleeps_between_calls():
    times = iter([0.0, 0.0, 0.0, 0.0, 0.01, 0.01])
    delays = []
    pacer = RequestPacer(10, clock=lambda: next(times), sleep=delays.append)
    pacer.wait()
    pacer.wait()
    assert delays == [pytest.approx(0.1)]


def _router_record(year, triggered=(), unavailable=(), price_return=0.1):
    financial = {"ttm_revenue_growth": 0.1, "gross_margin_change": 0.0, "ttm_operating_margin": 0.1, "ttm_operating_cash_flow": 10.0, "ttm_capex": 1.0, "ttm_free_cash_flow": 9.0, "free_cash_flow_positive_years": None, "roic": None, "net_debt_to_ebitda": 1.0, "interest_coverage": 5.0, "current_ratio": 1.5, "receivables_growth_minus_revenue_growth": 0.0, "inventory_growth_minus_revenue_growth": 0.0, "capex_to_revenue_change": 0.0, "earnings_surprise_zscore": 0.0, "earnings_volatility": 0.2, "operating_cash_flow_sign_changed": False}
    market = {"market_cap": 100.0, "free_cash_flow_yield": 0.09, "valuation_percentile": 0.5, "trailing_drawdown": -0.1, "sector_code": "SIC-35"}
    return {"availability": {"public_available_at": f"{year}-05-01T00:00:00Z"}, "route_labels": {"triggered_rules": list(triggered), "unavailable_rules": [{"rule_id": item, "missing_inputs": ["fixture"]} for item in unavailable], "conditional_edges": [], "ordered_models": []}, "financial_features": financial, "market_state": market, "exploratory_outcome": {"status": "exploratory_price_only", "raw_price_return": price_return}, "cik": "fixture", "issuer_name": "fixture", "ticker_at_collection": "FIX", "filing": {}}


def test_router_split_and_masked_targets_are_time_safe():
    assert split_for("2016-12-31") is None
    assert split_for("2017-01-01") == "train"
    record = _router_record(2018, triggered=("evidence_revision",), unavailable=("fragile_estimate",))
    assert rule_is_observed(record, "evidence_revision")
    assert not rule_is_observed(record, "fragile_estimate")
    y, mask, _returns, _return_mask, _clip, _mean_std = targets([record])
    assert y[0, ROUTE_RULES.index("evidence_revision")] == 1
    assert mask[0, ROUTE_RULES.index("fragile_estimate")] == 0


def test_router_preprocessor_uses_train_values_and_keeps_missingness():
    train = _router_record(2018)
    train["financial_features"]["ttm_operating_cash_flow"] = None
    preprocessor = RouterPreprocessor.fit([train])
    values, sectors, missing = preprocessor.transform([train])
    assert values.shape[1] == 2 * len(preprocessor.low)
    assert sectors.shape == (1,)
    assert missing.sum() >= 1


def test_router_decode_maps_rule_probability_to_explicit_graph():
    rules = load_rules(str(ROOT / "route_rules.yaml"))
    probabilities = {rule: 0.01 for rule in ROUTE_RULES}
    probabilities["balance_sheet_margin"] = 0.9
    decoded = decode_route(probabilities, {rule: 0.5 for rule in ROUTE_RULES}, rules)
    assert "mm-p1-safety-margin" in decoded["activated_models"]
    assert ["mm-p1-safety-margin", "mm-p1-expected-value"] in decoded["conditional_edges"]


def test_llm_teacher_packet_hides_rules_identity_and_future_outcome():
    record = _router_record(2020)
    record.update({"episode_id": "episode-fixture", "filing": {"form": "10-Q"}, "missing_reasons": {}, "source_quality": {}})
    packet = _packet(record)
    serialized = json.dumps(packet)
    assert not any(f'"{key}"' in serialized for key in FORBIDDEN_PACKET_KEYS)
    assert packet["episode_id"] == "episode-fixture"


def test_llm_teacher_label_validator_rejects_unknown_or_misordered_models():
    label = {
        "episode_id": "episode-fixture",
        "teacher_version": "gpt-5.6-sol-lollanet-teacher-v1",
        "selected_models": ["mm-p1-function-mapping", "mm-p1-expected-value", "unknown"],
        "ordered_models": ["mm-p1-expected-value", "mm-p1-function-mapping"],
        "confidence": 0.8,
        "evidence_fields": ["ttm_revenue_growth"],
        "uncertainty_reasons": [],
        "abstain": False,
        "model_rationales": {"mm-p1-function-mapping": "fixture", "mm-p1-expected-value": "fixture", "unknown": "fixture"},
        "cross_domain_mappings": [],
    }
    errors = validate_label(label, "episode-fixture")
    assert "selected_models contains duplicates or unknown models" in errors
    assert "ordered_models must be a permutation of selected_models" in errors
