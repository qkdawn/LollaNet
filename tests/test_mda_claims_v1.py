from __future__ import annotations

import hashlib
from pathlib import Path

from research.phase1.value_investing.download_sec_mda import normalize_html
from research.phase1.value_investing.mda_claims_v1 import (
    LocatorRecord,
    extract_episode,
    load_claim_types,
    stratified_sample,
    validate_claim_shape,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "research" / "phase1" / "microconcepts" / "qualitative_claim_types.yaml"


def _locator(tmp_path: Path, accession: str, body: bytes) -> LocatorRecord:
    raw_path = tmp_path / f"{accession}.html"
    raw_path.write_bytes(body)
    text = normalize_html(body)
    return LocatorRecord(
        accession=accession,
        available_at="2023-10-31T20:04:36.000Z",
        form="10-Q",
        section="MD&A",
        start_char=0,
        end_char=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        extractor_version="sec-mda-locator-v1",
        raw_html_sha256=hashlib.sha256(body).hexdigest(),
        raw_path=str(raw_path),
        episode_id=f"episode-{accession}",
        sic_division="SIC-35",
    )


def test_extract_episode_emits_only_taxonomy_claims_with_exact_evidence(tmp_path: Path) -> None:
    body = b"<html><body>Revenue increased 12% because demand was stronger. We expect liquidity will be sufficient over the next 12 months.</body></html>"
    locator = _locator(tmp_path, "0000000000-23-000001", body)
    claims = extract_episode(locator, load_claim_types(TAXONOMY))["claims"]
    assert {claim["claim_id"] for claim in claims} >= {"revenue_demand_change", "liquidity_outlook"}
    text = normalize_html(body)
    for claim in claims:
        assert text[claim["start_char"]:claim["end_char"]]
        assert claim["available_at"] == locator.available_at
        assert claim["text_sha256"] == locator.text_sha256
        assert not validate_claim_shape(claim, load_claim_types(TAXONOMY))


def test_extract_episode_abstains_instead_of_inventing_a_claim(tmp_path: Path) -> None:
    locator = _locator(tmp_path, "0000000000-23-000002", b"<html><body>This section describes the business.</body></html>")
    result = extract_episode(locator, load_claim_types(TAXONOMY))
    assert result["claims"] == []
    assert result["review_items"] == [{"accession": locator.accession, "section": "MD&A", "disposition": "abstain", "reason": "no_high_confidence_controlled_claim"}]


def test_stratified_sample_is_deterministic_and_keeps_every_stratum() -> None:
    locators = []
    for number in range(300):
        locators.append(LocatorRecord(
            accession=f"0000000000-23-{number:06d}", available_at=f"20{number % 3 + 20}-01-01T00:00:00.000Z",
            form="10-K" if number % 2 else "10-Q", section="MD&A", start_char=1, end_char=10,
            text_sha256="a" * 64, extractor_version="sec-mda-locator-v1", raw_html_sha256="b" * 64,
            raw_path="D:/not-read.html", episode_id=f"episode-{number}", sic_division=f"SIC-{number % 5:02d}",
        ))
    first = stratified_sample(locators, 300, "fixed")
    second = stratified_sample(locators, 300, "fixed")
    assert [item.accession for item in first] == [item.accession for item in second]
    assert {item.stratum for item in first} == {item.stratum for item in locators}


def test_claim_validator_rejects_route_or_outcome_fields(tmp_path: Path) -> None:
    locator = _locator(tmp_path, "0000000000-23-000003", b"<html><body>Revenue increased because demand was strong.</body></html>")
    claim = extract_episode(locator, load_claim_types(TAXONOMY))["claims"][0]
    claim["route_labels"] = {"forbidden": True}
    assert "forbidden raw text, route label, or outcome field present" in validate_claim_shape(claim, load_claim_types(TAXONOMY))

