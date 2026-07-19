import yaml

from research.phase1.microconcepts.generate_index import main as generate
from research.phase1.microconcepts.validate_microconcepts import CARD_ROOT, FEATURE_DICT, MODEL_ROOT, ROOT, frontmatter, load, validate


def cards():
    return [frontmatter(path) for path in sorted(CARD_ROOT.glob("*.md"))]


def test_microconcept_assets_validate() -> None:
    assert validate() == []


def test_markdown_is_the_only_card_source() -> None:
    assert len(cards()) == 11
    assert not [path for path in ROOT.glob("*.yaml") if path.name not in {"microconcept_schema.yaml", "microconcept_index.yaml", "qualitative_claim_types.yaml"}]
    assert all(card["status"] == "candidate" and card["version"] == "0.1.0" for card in cards())


def test_every_card_has_both_evidence_channels() -> None:
    for card in cards():
        assert card["quantitative_evidence"]["factors"]
        assert card["qualitative_evidence"]["claim_types"]
        assert card["qualitative_evidence"]["aggregation"]["missing_policy"] == "preserve_unknown"


def test_evidence_references_are_controlled() -> None:
    dictionary = load(FEATURE_DICT)
    factors = {item["name"] for group in ("financial_features", "market_state") for item in dictionary[group]}
    claims = {item["id"] for item in load(ROOT / "qualitative_claim_types.yaml")["claims"]}
    for card in cards():
        assert {item["id"] for item in card["quantitative_evidence"]["factors"]} <= factors
        assert set(card["qualitative_evidence"]["claim_types"]) <= claims


def test_all_phase1_models_have_a_microconcept_relation() -> None:
    model_ids = {load(path)["id"] for path in MODEL_ROOT.glob("*.yaml")}
    linked = {relation["model_id"] for card in cards() for relation in card["model_relations"]}
    assert linked == model_ids


def test_generated_artifacts_are_current() -> None:
    generate()
    index = yaml.safe_load((ROOT / "microconcept_index.yaml").read_text(encoding="utf-8"))
    assert index["source"] == "cards/*.md"
    assert {item["id"] for item in index["concepts"]} == {card["id"] for card in cards()}
    for name in ("MICROCONCEPT_INDEX.md", "EVIDENCE_CATALOG.md", "MODEL_MICROCONCEPT_GRAPH.md", "MICROCONCEPT_EVIDENCE_GRAPH.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert text.startswith("---\n")
    assert "```mermaid" in (ROOT / "MODEL_MICROCONCEPT_GRAPH.md").read_text(encoding="utf-8")
    assert "```mermaid" in (ROOT / "MICROCONCEPT_EVIDENCE_GRAPH.md").read_text(encoding="utf-8")


def test_anonymous_fixture_has_no_raw_text_or_outcome() -> None:
    fixture = load(ROOT / "fixtures" / "microconcept_evidence_fixture.yaml")
    serialized = yaml.safe_dump(fixture)
    assert "raw_mda_text:" not in serialized
    assert "outcome:" not in serialized
    assert fixture["available_at"] <= fixture["decision_at"]
