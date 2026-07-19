"""Validate Markdown microconcept cards and their evidence contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CARD_ROOT = ROOT / "cards"
MODEL_ROOT = ROOT.parent / "models"
FEATURE_DICT = ROOT.parent / "value_investing" / "feature_dictionary.yaml"
CLAIM_TYPES = ROOT / "qualitative_claim_types.yaml"
FORBIDDEN = {"raw_mda_text", "outcome", "future_return", "return_12m", "delisting_return"}


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    _, payload, _ = text.split("---", 2)
    value = yaml.safe_load(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: frontmatter must be a mapping")
    return value


def load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return (set(value) & FORBIDDEN) | set().union(*(forbidden_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(forbidden_keys(item) for item in value), set())
    return set()


def validate(root: Path = ROOT) -> list[str]:
    schema = load(root / "microconcept_schema.yaml")
    cards = [(path, frontmatter(path)) for path in sorted((root / "cards").glob("*.md"))]
    errors: list[str] = []
    ids: set[str] = set()
    model_ids = {load(path)["id"] for path in sorted(MODEL_ROOT.glob("*.yaml"))}
    dictionary = load(FEATURE_DICT)
    factors = {item["name"] for group in ("financial_features", "market_state") for item in dictionary[group]}
    claims = {item["id"]: item for item in load(CLAIM_TYPES)["claims"]}

    for path, card in cards:
        label = path.name
        missing = set(schema["required_frontmatter"]) - set(card)
        if missing:
            errors.append(f"{label}: missing frontmatter {sorted(missing)}")
        concept_id = card.get("id")
        if not isinstance(concept_id, str) or not concept_id.isascii() or not concept_id.replace("_", "").isalnum():
            errors.append(f"{label}: id must be ASCII snake_case")
        elif concept_id in ids:
            errors.append(f"{label}: duplicate id {concept_id}")
        else:
            ids.add(concept_id)
        if card.get("type") != "microconcept" or card.get("semantic_level") != "microconcept":
            errors.append(f"{label}: invalid card type or semantic level")
        if card.get("version") != "0.1.0" or card.get("status") != "candidate":
            errors.append(f"{label}: v0 cards must be candidate 0.1.0")
        if card.get("family") not in schema["families"]:
            errors.append(f"{label}: invalid family")
        for field in ("definition", "mechanism", "evidence_boundary"):
            if not isinstance(card.get(field), str) or not card[field].strip():
                errors.append(f"{label}: {field} must be non-empty text")
        quantitative = card.get("quantitative_evidence", {})
        factor_rows = quantitative.get("factors") if isinstance(quantitative, dict) else None
        if not isinstance(factor_rows, list) or not factor_rows:
            errors.append(f"{label}: quantitative_evidence.factors must be non-empty")
        else:
            for row in factor_rows:
                if not isinstance(row, dict) or not {"id", "role", "transformation", "expected_direction"} <= set(row):
                    errors.append(f"{label}: malformed quantitative factor row")
                elif row["id"] not in factors:
                    errors.append(f"{label}: unknown quantitative factor {row['id']}")
        qualitative = card.get("qualitative_evidence", {})
        claim_rows = qualitative.get("claim_types") if isinstance(qualitative, dict) else None
        if not isinstance(claim_rows, list) or not claim_rows:
            errors.append(f"{label}: qualitative_evidence.claim_types must be non-empty")
        else:
            for claim_id in claim_rows:
                if claim_id not in claims:
                    errors.append(f"{label}: unknown qualitative claim {claim_id}")
            aggregation = qualitative.get("aggregation", {})
            if aggregation.get("direction") != "signed_claims" or aggregation.get("missing_policy") != "preserve_unknown":
                errors.append(f"{label}: qualitative aggregation must preserve signed unknowns")
        interface = card.get("output_interface")
        if not isinstance(interface, dict) or set(interface) != {"presence", "intensity", "direction", "confidence"}:
            errors.append(f"{label}: output_interface must expose four standard outputs")
        relations = card.get("model_relations")
        if not isinstance(relations, list) or not relations:
            errors.append(f"{label}: model_relations must be non-empty")
        else:
            relation_ids = set()
            for relation in relations:
                if not isinstance(relation, dict) or not {"model_id", "relation", "rationale"} <= set(relation):
                    errors.append(f"{label}: malformed model relation")
                    continue
                relation_ids.add(relation["model_id"])
                if relation["model_id"] not in model_ids:
                    errors.append(f"{label}: unknown model {relation['model_id']}")
                if relation["relation"] not in schema["relation_types"]:
                    errors.append(f"{label}: invalid relation type {relation['relation']}")
            if len(relation_ids) != len(relations):
                errors.append(f"{label}: duplicate model relation")
        if forbidden_keys(card):
            errors.append(f"{label}: contains forbidden future, outcome, or raw-text field")

    if len(cards) != 11:
        errors.append(f"expected 11 Markdown cards, found {len(cards)}")
    if set(ids) != {path.stem for path, _ in cards}:
        errors.append("card filename stems must match card IDs")
    if cards and len(model_ids) != 20:
        errors.append("Phase 1 model asset count is not 20")
    linked = {relation["model_id"] for _, card in cards for relation in card.get("model_relations", [])}
    missing_models = model_ids - linked
    if missing_models:
        errors.append(f"models without microconcept relation: {sorted(missing_models)}")

    fixture = root / "fixtures" / "microconcept_evidence_fixture.yaml"
    if fixture.exists():
        errors.extend(validate_fixture(fixture, factors, set(claims)))
    return errors


def validate_fixture(path: Path, factors: set[str], claims: set[str]) -> list[str]:
    payload = load(path)
    errors: list[str] = []
    available_at = payload.get("available_at")
    decision_at = payload.get("decision_at")
    if not available_at or not decision_at or available_at > decision_at:
        errors.append("fixture: evidence availability must not be after decision time")
    if forbidden_keys(payload):
        errors.append("fixture: contains forbidden outcome or raw text")
    for field in payload.get("quantitative_factors", {}):
        if field not in factors:
            errors.append(f"fixture: unknown factor {field}")
    for evidence in payload.get("qualitative_evidence", []):
        if evidence.get("claim_id") not in claims:
            errors.append(f"fixture: unknown claim {evidence.get('claim_id')}")
        source = evidence.get("source", {})
        required = {"accession", "section", "start_char", "end_char", "text_sha256", "extractor_version"}
        if not required <= set(source):
            errors.append("fixture: qualitative evidence is missing locator fields")
    return errors


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors = validate()
    if errors:
        print("Microconcept validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("Microconcept Markdown layer is valid (11 candidate cards).")


if __name__ == "__main__":
    main()
