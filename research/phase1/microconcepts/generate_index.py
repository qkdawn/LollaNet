"""Generate indexes, evidence catalog, and Mermaid views from Markdown cards."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import yaml

try:
    from .validate_microconcepts import CARD_ROOT, CLAIM_TYPES, FEATURE_DICT, MODEL_ROOT, ROOT, frontmatter, load, validate
except ImportError:
    from validate_microconcepts import CARD_ROOT, CLAIM_TYPES, FEATURE_DICT, MODEL_ROOT, ROOT, frontmatter, load, validate


COLORS = {
    "model": "fill:#dbeafe,stroke:#2563eb,color:#172554",
    "microconcept": "fill:#dcfce7,stroke:#16a34a,color:#052e16",
    "factor": "fill:#ffedd5,stroke:#ea580c,color:#431407",
    "claim": "fill:#f3e8ff,stroke:#9333ea,color:#3b0764",
}


def node_id(prefix: str, value: str) -> str:
    return prefix + re.sub(r"[^A-Za-z0-9_]", "_", value)


def mermaid(lines: list[str]) -> str:
    styles = [f"classDef {kind} {style};" for kind, style in COLORS.items()]
    return "```mermaid\nflowchart LR\n" + "\n".join("  " + line for line in [*lines, *styles]) + "\n```"


def main() -> None:
    errors = validate()
    if errors:
        raise SystemExit("Cannot generate artifacts:\n" + "\n".join(errors))
    cards = [frontmatter(path) for path in sorted(CARD_ROOT.glob("*.md"))]
    models = {payload["id"]: payload for path in MODEL_ROOT.glob("*.yaml") if (payload := load(path))}
    dictionary = load(FEATURE_DICT)
    factors = {item["name"]: item for group in ("financial_features", "market_state") for item in dictionary[group]}
    claims = {item["id"]: item for item in load(CLAIM_TYPES)["claims"]}
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    edges = []
    concepts = []
    for card in cards:
        quantitative = [row["id"] for row in card["quantitative_evidence"]["factors"]]
        qualitative = list(card["qualitative_evidence"]["claim_types"])
        model_ids = [row["model_id"] for row in card["model_relations"]]
        concepts.append({"id": card["id"], "family": card["family"], "quantitative_factors": quantitative, "qualitative_claims": qualitative, "models": model_ids})
        for relation in card["model_relations"]:
            by_model[relation["model_id"]].append({"microconcept_id": card["id"], "relation": relation["relation"]})
            edges.append({"source": card["id"], "target": relation["model_id"], "relation": relation["relation"], "rationale": relation["rationale"]})
        for row in card["quantitative_evidence"]["factors"]:
            edges.append({"source": card["id"], "target": row["id"], "relation": "grounded_by", "rationale": f"{row['role']} via {row['transformation']}"})
        for claim_id in qualitative:
            edges.append({"source": card["id"], "target": claim_id, "relation": "modulated_by", "rationale": claims[claim_id]["definition"]})
    index = {
        "version": "0.1.0", "status": "read_only_generated", "source": "cards/*.md",
        "concepts": concepts,
        "by_model": [{"model_id": model_id, "microconcepts": sorted(rows, key=lambda row: row["microconcept_id"])} for model_id, rows in sorted(by_model.items())],
        "edges": edges,
    }
    (ROOT / "microconcept_index.yaml").write_text(yaml.safe_dump(index, allow_unicode=True, sort_keys=False), encoding="utf-8")
    generate_catalog(factors, claims)
    generate_index(cards)
    generate_model_graph(cards, models)
    generate_evidence_graph(cards, models, factors, claims)
    print("Generated microconcept index, evidence catalog, and Mermaid views")


def generate_catalog(factors: dict, claims: dict) -> None:
    lines = ["---", "type: research_evidence_catalog", "title: Phase 1 Evidence Catalog", "status: candidate", "generated_by: research/phase1/microconcepts/generate_index.py", "---", "", "# Evidence Catalog", "", "此页由字段字典和受控命题目录生成，只用于导航。", "", "## Quantitative Factors"]
    for name, item in sorted(factors.items()):
        lines += ["", f"### Factor: {name}", "", f"- Unit: `{item['unit']}`", f"- Source: {item['source']}", f"- Formula: {item['formula']}"]
    lines += ["", "## Qualitative Claims"]
    for name, item in sorted(claims.items()):
        lines += ["", f"### Claim: {name}", "", f"- Family: `{item['family']}`", f"- Definition: {item['definition']}", "- Locator: accession, section, start/end offset, SHA-256, extractor version"]
    (ROOT / "EVIDENCE_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_index(cards: list[dict]) -> None:
    lines = ["---", "type: research_index", "title: Phase 1 Microconcept Index", "status: candidate", "generated_by: research/phase1/microconcepts/generate_index.py", "---", "", "# Microconcept Index", "", "Markdown 卡是唯一事实源；本页为生成导航。", "", "| Microconcept | Family | Quantitative | Qualitative | Models |", "| --- | --- | --- | --- | --- |"]
    for card in cards:
        factors = ", ".join(row["id"] for row in card["quantitative_evidence"]["factors"])
        claims = ", ".join(card["qualitative_evidence"]["claim_types"])
        models = ", ".join(row["model_id"] for row in card["model_relations"])
        lines.append(f"| [[{card['id']}]] | {card['family']} | {factors} | {claims} | {models} |")
    (ROOT / "MICROCONCEPT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_model_graph(cards: list[dict], models: dict) -> None:
    overview = []
    for card in cards:
        cid = node_id("c_", card["id"])
        overview.append(f'{cid}["{card["id"]}"]:::microconcept')
        for relation in card["model_relations"]:
            mid = node_id("m_", relation["model_id"])
            overview += [f'{mid}["{models[relation["model_id"]]["name"]}"]:::model', f'{cid} -->|{relation["relation"]}| {mid}']
    lines = ["---", "type: research_map", "title: Model Microconcept Graph", "status: candidate", "generated_by: research/phase1/microconcepts/generate_index.py", "---", "", "# Model Microconcept Graph", "", "## Overview", "", mermaid(overview)]
    for model_id, model in sorted(models.items()):
        local = [f'{node_id("m_", model_id)}["{model["name"]}"]:::model']
        for card in cards:
            for relation in card["model_relations"]:
                if relation["model_id"] == model_id:
                    local += [f'{node_id("c_", card["id"])}["{card["id"]}"]:::microconcept', f'{node_id("c_", card["id"])} -->|{relation["relation"]}| {node_id("m_", model_id)}']
        lines += ["", f"## {model['name']} (`{model_id}`)", "", mermaid(local)]
    (ROOT / "MODEL_MICROCONCEPT_GRAPH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_evidence_graph(cards: list[dict], models: dict, factors: dict, claims: dict) -> None:
    lines = ["---", "type: research_map", "title: Microconcept Evidence Graph", "status: candidate", "generated_by: research/phase1/microconcepts/generate_index.py", "---", "", "# Microconcept Evidence Graph", "", "总览按微概念分组；下方局部图用于审阅证据链。"]
    overview = []
    for card in cards:
        cid = node_id("c_", card["id"])
        overview.append(f'{cid}["{card["id"]}"]:::microconcept')
        for row in card["quantitative_evidence"]["factors"]:
            fid = node_id("f_", row["id"])
            overview += [f'{fid}["{row["id"]}"]:::factor', f'{fid} --> {cid}']
        for claim_id in card["qualitative_evidence"]["claim_types"]:
            qid = node_id("q_", claim_id)
            overview += [f'{qid}["{claim_id}"]:::claim', f'{qid} --> {cid}']
        for relation in card["model_relations"]:
            mid = node_id("m_", relation["model_id"])
            overview += [f'{mid}["{models[relation["model_id"]]["name"]}"]:::model', f'{cid} -->|{relation["relation"]}| {mid}']
    lines += ["", "## Overview", "", mermaid(overview)]
    for card in cards:
        cid = node_id("c_", card["id"])
        local = [f'{cid}["{card["id"]}"]:::microconcept']
        for row in card["quantitative_evidence"]["factors"]:
            local += [f'{node_id("f_", row["id"])}["{row["id"]}"]:::factor', f'{node_id("f_", row["id"])} -->|{row["role"]}| {cid}']
        for claim_id in card["qualitative_evidence"]["claim_types"]:
            local += [f'{node_id("q_", claim_id)}["{claim_id}"]:::claim', f'{node_id("q_", claim_id)} -->|modulates| {cid}']
        for relation in card["model_relations"]:
            mid = node_id("m_", relation["model_id"])
            local += [f'{mid}["{models[relation["model_id"]]["name"]}"]:::model', f'{cid} -->|{relation["relation"]}| {mid}']
        lines += ["", f"## {card['id']}", "", mermaid(local)]
    (ROOT / "MICROCONCEPT_EVIDENCE_GRAPH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
