"""Deterministic, evidence-bound construction of the real MD&A claims v1 pilot.

The module intentionally reads only filing identifiers, availability dates, form,
SIC, locator metadata, and the SEC filing text selected by the existing locator.
It never uses issuer identity, financial features, market state, route labels, or
future outcomes as extraction inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from research.phase1.value_investing.download_sec_mda import normalize_html


CLAIM_DATASET_VERSION = "mda-claims-v1.0.0"
CLAIM_EXTRACTOR_VERSION = "mda-claims-rule-extractor-v1.0.1"
LABEL_SOURCE = "deterministic_mda_rule_label_v1"
DEFAULT_SEED = "mda-claims-v1-stratified-20260719"
REQUIRED_CLAIM_FIELDS = {
    "claim_key", "claim_id", "polarity", "time_horizon", "strength", "confidence",
    "accession", "available_at", "section", "start_char", "end_char", "text_sha256", "evidence_sha256",
    "extractor_version", "claim_extractor_version", "label_source",
}
FORBIDDEN_INPUT_OR_OUTPUT_KEYS = {
    "route_labels", "exploratory_outcome", "market_state", "financial_features", "issuer_name",
    "ticker_at_collection", "security_type", "training_status", "raw_mda_text", "text",
}


@dataclass(frozen=True)
class ClaimType:
    claim_id: str
    polarities: frozenset[str]
    horizons: frozenset[str]
    strengths: frozenset[str]


@dataclass(frozen=True)
class LocatorRecord:
    accession: str
    available_at: str
    form: str
    section: str
    start_char: int
    end_char: int
    text_sha256: str
    extractor_version: str
    raw_html_sha256: str
    raw_path: str
    episode_id: str
    sic_division: str

    @property
    def year(self) -> str:
        return self.available_at[:4]

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.year, self.form, self.sic_division)

    def fingerprint(self) -> str:
        return stable_hash({
            "accession": self.accession, "available_at": self.available_at, "form": self.form,
            "section": self.section, "start_char": self.start_char, "end_char": self.end_char,
            "text_sha256": self.text_sha256, "extractor_version": self.extractor_version,
            "raw_html_sha256": self.raw_html_sha256, "raw_path": self.raw_path,
            "episode_id": self.episode_id, "sic_division": self.sic_division,
        })


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def load_claim_types(path: Path) -> dict[str, ClaimType]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    result: dict[str, ClaimType] = {}
    for item in payload.get("claims", []):
        result[item["id"]] = ClaimType(item["id"], frozenset(item["polarity"]), frozenset(item["time_horizon"]), frozenset(item["strength"]))
    if not result:
        raise ValueError(f"No claim types found in {path}")
    return result


def sic_division(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return f"SIC-{digits[:2]}" if len(digits) >= 2 else "SIC-unknown"


def load_episode_sic(path: Path) -> dict[str, str]:
    """Read only episode identity and SIC; deliberately ignore all label/outcome fields."""
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str):
                raise ValueError(f"{path}:{line_number} has no episode_id")
            result[episode_id] = sic_division(record.get("sic"))
    return result


def load_locators(index_path: Path, episode_sics: dict[str, str]) -> tuple[list[LocatorRecord], Counter[tuple[str | None, str | None]], int]:
    high: list[LocatorRecord] = []
    status_counts: Counter[tuple[str | None, str | None]] = Counter()
    total = 0
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            status_counts[(item.get("status"), item.get("locator_quality"))] += 1
            if not (item.get("status") == "located" and item.get("locator_quality") == "high"):
                continue
            required = {"accession", "available_at", "form", "section", "start_char", "end_char", "text_sha256", "extractor_version", "raw_html_sha256", "raw_path", "episode_id"}
            absent = required - set(item)
            if absent:
                raise ValueError(f"{index_path}:{line_number} lacks {sorted(absent)}")
            episode_id = item["episode_id"]
            if episode_id not in episode_sics:
                raise ValueError(f"{index_path}:{line_number}: episode_id {episode_id!r} is missing from candidates")
            high.append(LocatorRecord(
                accession=item["accession"], available_at=item["available_at"], form=item["form"], section=item["section"],
                start_char=int(item["start_char"]), end_char=int(item["end_char"]), text_sha256=item["text_sha256"],
                extractor_version=item["extractor_version"], raw_html_sha256=item["raw_html_sha256"], raw_path=item["raw_path"],
                episode_id=episode_id, sic_division=episode_sics[episode_id],
            ))
    return high, status_counts, total


def _rank(seed: str, locator: LocatorRecord) -> str:
    return stable_hash({"seed": seed, "accession": locator.accession, "episode_id": locator.episode_id})


def stratified_sample(locators: list[LocatorRecord], sample_size: int, seed: str) -> list[LocatorRecord]:
    """Deterministically cover years, forms, industries, and year-form pairs.

    A complete year/form/industry Cartesian cover has 929 observed cells in the
    current universe, which is incompatible with the required 300–500 pilot
    size. This design first guarantees each marginal and each observed year-form
    cell, then fills remaining capacity proportionally across the joint strata.
    """
    if not 300 <= sample_size <= 500:
        raise ValueError("sample_size must be between 300 and 500 inclusive")
    if sample_size > len(locators):
        raise ValueError("sample_size exceeds eligible high-confidence MD&A locators")
    groups: dict[tuple[str, str, str], list[LocatorRecord]] = defaultdict(list)
    for locator in locators:
        groups[locator.stratum].append(locator)
    ordered_groups = sorted(groups)
    for group in ordered_groups:
        groups[group].sort(key=lambda item: (_rank(seed, item), item.accession))

    selected: list[LocatorRecord] = []
    selected_keys: set[tuple[str, str]] = set()
    required = {
        *( ("year", item.year) for item in locators ),
        *( ("form", item.form) for item in locators ),
        *( ("sic_division", item.sic_division) for item in locators ),
        *( ("year_form", f"{item.year}|{item.form}") for item in locators ),
    }
    covered: set[tuple[str, str]] = set()
    ranked = sorted(locators, key=lambda item: (_rank(seed, item), item.accession))
    while covered != required:
        candidates = []
        for locator in ranked:
            key = (locator.accession, locator.episode_id)
            if key in selected_keys:
                continue
            gains = {
                ("year", locator.year),
                ("form", locator.form),
                ("sic_division", locator.sic_division),
                ("year_form", f"{locator.year}|{locator.form}"),
            } - covered
            if gains:
                candidates.append((len(gains), locator))
        if not candidates:
            raise ValueError("could not satisfy declared coverage dimensions")
        _, chosen = max(candidates, key=lambda item: (item[0], _rank(seed, item[1]), item[1].accession))
        selected.append(chosen)
        selected_keys.add((chosen.accession, chosen.episode_id))
        covered.update({
            ("year", chosen.year), ("form", chosen.form), ("sic_division", chosen.sic_division),
            ("year_form", f"{chosen.year}|{chosen.form}"),
        })
    if len(selected) > sample_size:
        raise ValueError(f"coverage requires {len(selected)} records, exceeding requested sample size {sample_size}")

    by_group_selected = Counter(item.stratum for item in selected)
    total = len(locators)
    quotas = {group: sample_size * len(groups[group]) / total for group in ordered_groups}
    group_positions = {group: 0 for group in ordered_groups}
    for group in ordered_groups:
        while group_positions[group] < len(groups[group]) and (groups[group][group_positions[group]].accession, groups[group][group_positions[group]].episode_id) in selected_keys:
            group_positions[group] += 1
    while len(selected) < sample_size:
        candidates = [group for group in ordered_groups if group_positions[group] < len(groups[group])]
        if not candidates:
            raise ValueError("not enough capacity to fill requested sample")
        group = max(candidates, key=lambda item: (quotas[item] - by_group_selected[item], len(groups[item]) - group_positions[item], stable_hash({"seed": seed, "stratum": item})))
        chosen = groups[group][group_positions[group]]
        selected.append(chosen)
        selected_keys.add((chosen.accession, chosen.episode_id))
        by_group_selected[group] += 1
        group_positions[group] += 1
        while group_positions[group] < len(groups[group]) and (groups[group][group_positions[group]].accession, groups[group][group_positions[group]].episode_id) in selected_keys:
            group_positions[group] += 1
    return sorted(selected, key=lambda item: (item.available_at, item.form, item.sic_division, item.accession))


def locator_to_sample_row(locator: LocatorRecord) -> dict[str, Any]:
    return {
        "accession": locator.accession, "available_at": locator.available_at, "episode_id": locator.episode_id,
        "form": locator.form, "section": locator.section, "start_char": locator.start_char, "end_char": locator.end_char,
        "text_sha256": locator.text_sha256, "extractor_version": locator.extractor_version,
        "raw_html_sha256": locator.raw_html_sha256, "raw_path": locator.raw_path, "sic_division": locator.sic_division,
        "sampling_stratum": {"year": locator.year, "form": locator.form, "sic_division": locator.sic_division},
        "source_fingerprint": locator.fingerprint(),
    }


def locator_from_sample_row(row: dict[str, Any]) -> LocatorRecord:
    return LocatorRecord(
        accession=row["accession"], available_at=row["available_at"], episode_id=row["episode_id"], form=row["form"], section=row["section"],
        start_char=int(row["start_char"]), end_char=int(row["end_char"]), text_sha256=row["text_sha256"], extractor_version=row["extractor_version"],
        raw_html_sha256=row["raw_html_sha256"], raw_path=row["raw_path"], sic_division=row["sic_division"],
    )


def _sentence_spans(section: str, absolute_start: int) -> Iterable[tuple[int, int, str]]:
    """Yield conservative sentence spans in normalized MD&A text without copying text to outputs."""
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|(?<=;)\s+(?=[A-Z])", section):
        yield from _clean_sentence(section, start, match.start(), absolute_start)
        start = match.end()
    yield from _clean_sentence(section, start, len(section), absolute_start)


def _clean_sentence(section: str, start: int, end: int, absolute_start: int) -> Iterable[tuple[int, int, str]]:
    while start < end and section[start].isspace():
        start += 1
    while end > start and section[end - 1].isspace():
        end -= 1
    if not start < end:
        return
    sentence = section[start:end]
    if len(sentence) > 1_200 or len(sentence) < 24:
        return
    yield absolute_start + start, absolute_start + end, sentence


POSITIVE_WORDS = re.compile(r"\b(increas(?:e|ed|es|ing)|improv(?:e|ed|ement|ing)|higher|growth|grew|expand(?:ed|ing)?|strong(?:er)?|favorable|sufficient|adequate|benefit)\b", re.I)
NEGATIVE_WORDS = re.compile(r"\b(decreas(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|lower|weak(?:er|ness)?|pressure|compress(?:ion|ed|ing)?|adverse|unfavorable|shortfall|insufficient|constraint|constrain(?:ed|ts|ing)?|disrupt(?:ion|ed|ions)|reduc(?:e|ed|tion|ing)|breach|default|violation)\b", re.I)
UNCERTAIN_WORDS = re.compile(r"\b(may|might|could|uncertain|uncertainty|possible|potential)\b", re.I)
STRONG_WORDS = re.compile(r"\b(must|will|required|required to|material|significant|substantial)\b|\$|\b\d+(?:\.\d+)?%\b", re.I)
MEDIUM_WORDS = re.compile(r"\b(expect|expects|expected|anticipate|anticipates|believe|believes|plan|plans|intend|should)\b", re.I)
NEUTRAL_NEGATION = re.compile(r"\b(?:do not|does not|did not|not|no)\s+(?:expect|anticipate|believe|plan|intend)\b", re.I)


def classify_polarity(sentence: str, default: str = "unknown") -> str:
    if NEUTRAL_NEGATION.search(sentence):
        return "unknown"
    positive, negative = bool(POSITIVE_WORDS.search(sentence)), bool(NEGATIVE_WORDS.search(sentence))
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return default


def classify_horizon(sentence: str) -> str:
    if re.search(r"\b(long[- ]term|over the next several years|over several years|multi[- ]year)\b", sentence, re.I):
        return "long_term"
    if re.search(r"\b(next 12 months|within 12 months|next fiscal year|next year|coming year|next quarter|near[- ]term)\b", sentence, re.I):
        return "near_term"
    if re.search(r"\b(over the next (?:two|three|few) years|medium[- ]term)\b", sentence, re.I):
        return "medium_term"
    return "unspecified"


def classify_strength(sentence: str) -> str:
    if UNCERTAIN_WORDS.search(sentence):
        return "weak"
    if STRONG_WORDS.search(sentence):
        return "strong"
    if MEDIUM_WORDS.search(sentence):
        return "medium"
    return "unknown"


def _has_all(sentence: str, *patterns: str) -> bool:
    return all(re.search(pattern, sentence, re.I) for pattern in patterns)


def candidate_type_ids(sentence: str) -> list[str]:
    """Return only claim types supported by explicit language in one evidence span."""
    candidates: list[str] = []
    if _has_all(sentence, r"\b(revenue|net sales|sales|demand|orders?)\b", r"\b(increased?|decreased?|declined?|grew|growth|higher|lower|weak(?:ened|er)?|strong(?:er)?)\b"):
        candidates.append("revenue_demand_change")
    if _has_all(sentence, r"\b(gross|operating|profit) margins?\b", r"\b(increased?|decreased?|declined?|improved?|higher|lower|pressure|compress(?:ion|ed|ing)?)\b"):
        candidates.append("margin_pressure")
    if _has_all(sentence, r"\b(accounts receivable|receivables|inventor(?:y|ies)|working capital)\b", r"\b(increased?|decreased?|declined?|higher|lower|change[ds]?|grew|growth)\b"):
        candidates.append("working_capital_change")
    if _has_all(sentence, r"\bliquidity\b", r"\b(expect|anticipate|believe|sufficient|adequate|available|will|should)\b"):
        candidates.append("liquidity_outlook")
    if _has_all(sentence, r"\b(refinanc(?:e|ing)|financing|credit facilities?|borrowings?|debt)\b", r"\b(need|depend(?:ent|ency)?|rely|require(?:d|ment)?|must)\b"):
        candidates.append("refinancing_dependency")
    if _has_all(sentence, r"\bcovenants?\b", r"\b(breach|violation|default|compliance|waiv(?:e|er)|risk|unable|failure)\b"):
        candidates.append("covenant_risk")
    if _has_all(sentence, r"\b(capital expenditures?|capex|share repurchases?|repurchase program|dividends?|acquisitions?|mergers?|strategic investments?)\b", r"\b(increased?|decreased?|reduc(?:e|ed|tion|ing)|suspend(?:ed|ing)?|plan|expect|intend|will|change[ds]?)\b"):
        candidates.append("capital_allocation_change")
    if _has_all(sentence, r"\b(return on (?:invested )?capital|return on investment|disciplined investment|investment returns?)\b", r"\b(return|returns?|improve|improved|expect|target|disciplined)\b"):
        candidates.append("reinvestment_quality")
    if _has_all(sentence, r"\b(supply chain|operational risk|execution risk|labor (?:shortage|availability)|cyber(?:security)?|inflation|disruption)\b", r"\b(risk|could|may|might|adverse|impact|disrupt|shortage|constraint|pressure)\b"):
        candidates.append("operational_risk")
    if _has_all(sentence, r"\b(expect|anticipate|believe|outlook|forecast)\b", r"\b(demand|sales|revenue|market|orders?|growth)\b"):
        candidates.append("demand_outlook")
    if _has_all(sentence, r"\b(guidance|outlook)\b", r"\b(raise[ds]?|increas(?:e|ed|es|ing)|lower(?:ed|ing)?|reduc(?:e|ed|tion|ing)|withdraw(?:n|al)?|revis(?:e|ed|ion|ing))\b"):
        candidates.append("guidance_revision")
    return candidates


def claim_confidence(sentence: str, claim_id: str, polarity: str) -> float:
    confidence = 0.92
    if polarity in {"unknown", "mixed"}:
        confidence -= 0.05
    if claim_id in {"capital_allocation_change", "working_capital_change", "reinvestment_quality"}:
        confidence -= 0.02
    if len(sentence) > 800:
        confidence -= 0.05
    return round(max(0.0, confidence), 2)


def build_claim(locator: LocatorRecord, claim_id: str, start: int, end: int, text: str) -> dict[str, Any]:
    default = "negative" if claim_id in {"covenant_risk", "operational_risk", "refinancing_dependency"} else "unknown"
    polarity = classify_polarity(text, default)
    if claim_id in {"capital_allocation_change", "working_capital_change"} and polarity == "mixed":
        polarity = "unknown"
    claim = {
        "claim_id": claim_id, "polarity": polarity, "time_horizon": classify_horizon(text), "strength": classify_strength(text),
        "confidence": claim_confidence(text, claim_id, polarity), "accession": locator.accession, "available_at": locator.available_at, "section": locator.section,
        "start_char": start, "end_char": end, "text_sha256": locator.text_sha256, "evidence_sha256": sha256_text(text),
        "extractor_version": locator.extractor_version, "claim_extractor_version": CLAIM_EXTRACTOR_VERSION, "label_source": LABEL_SOURCE,
    }
    claim["claim_key"] = stable_hash({"accession": locator.accession, "claim_id": claim_id, "start_char": start, "end_char": end, "evidence_sha256": claim["evidence_sha256"]})
    return claim


def extract_episode(locator: LocatorRecord, claim_types: dict[str, ClaimType]) -> dict[str, Any]:
    raw_path = Path(locator.raw_path)
    def abstain(reason: str) -> dict[str, Any]:
        return {"source_fingerprint": locator.fingerprint(), "claim_extractor_version": CLAIM_EXTRACTOR_VERSION, "claims": [], "review_items": [{"accession": locator.accession, "section": locator.section, "disposition": "abstain", "reason": reason}]}
    if not raw_path.exists():
        return abstain("raw_filing_missing")
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != locator.raw_html_sha256:
        return abstain("raw_html_sha256_mismatch")
    text = normalize_html(raw)
    if not (0 <= locator.start_char < locator.end_char <= len(text)):
        return abstain("locator_bounds_invalid")
    section = text[locator.start_char:locator.end_char]
    if sha256_text(section) != locator.text_sha256:
        return abstain("locator_text_sha256_mismatch")
    chosen: dict[str, dict[str, Any]] = {}
    review_items: list[dict[str, Any]] = []
    for start, end, sentence in _sentence_spans(section, locator.start_char):
        for claim_id in candidate_type_ids(sentence):
            if claim_id not in claim_types:
                raise ValueError(f"Classifier emitted non-taxonomy claim id {claim_id}")
            claim = build_claim(locator, claim_id, start, end, sentence)
            if claim["confidence"] < 0.80:
                review_items.append({"accession": locator.accession, "section": locator.section, "disposition": "review", "reason": "low_mapping_confidence", "candidate_claim_id": claim_id, "start_char": start, "end_char": end, "text_sha256": locator.text_sha256, "evidence_sha256": claim["evidence_sha256"], "confidence": claim["confidence"]})
                continue
            current = chosen.get(claim_id)
            if current is None or (claim["confidence"], -claim["start_char"]) > (current["confidence"], -current["start_char"]):
                chosen[claim_id] = claim
    if not chosen:
        review_items.append({"accession": locator.accession, "section": locator.section, "disposition": "abstain", "reason": "no_high_confidence_controlled_claim"})
    for claim in chosen.values():
        if claim["polarity"] in {"mixed", "unknown"}:
            review_items.append({
                "accession": locator.accession,
                "section": locator.section,
                "disposition": "review",
                "reason": "mixed_polarity_evidence" if claim["polarity"] == "mixed" else "polarity_unknown_requires_human_review",
                "claim_key": claim["claim_key"],
                "candidate_claim_id": claim["claim_id"],
                "start_char": claim["start_char"],
                "end_char": claim["end_char"],
                "text_sha256": claim["text_sha256"],
                "evidence_sha256": claim["evidence_sha256"],
                "confidence": claim["confidence"],
            })
    return {"source_fingerprint": locator.fingerprint(), "claim_extractor_version": CLAIM_EXTRACTOR_VERSION, "claims": sorted(chosen.values(), key=lambda item: (item["start_char"], item["claim_id"])), "review_items": review_items}


def source_distribution(sample: Iterable[LocatorRecord]) -> dict[str, dict[str, int]]:
    sample = list(sample)
    return {"year": dict(sorted(Counter(item.year for item in sample).items())), "form": dict(sorted(Counter(item.form for item in sample).items())), "sic_division": dict(sorted(Counter(item.sic_division for item in sample).items())), "year_form": dict(sorted(Counter(f"{item.year}|{item.form}" for item in sample).items()))}


def claims_distribution(claims: Iterable[dict[str, Any]]) -> dict[str, Any]:
    claims = list(claims)
    return {"claim_id": dict(sorted(Counter(item["claim_id"] for item in claims).items())), "polarity": dict(sorted(Counter(item["polarity"] for item in claims).items())), "time_horizon": dict(sorted(Counter(item["time_horizon"] for item in claims).items())), "strength": dict(sorted(Counter(item["strength"] for item in claims).items())), "label_source": dict(sorted(Counter(item["label_source"] for item in claims).items())), "mean_confidence": round(sum(float(item["confidence"]) for item in claims) / len(claims), 4) if claims else None}


def validate_claim_shape(claim: dict[str, Any], claim_types: dict[str, ClaimType]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_CLAIM_FIELDS - set(claim)
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    claim_type = claim_types.get(claim["claim_id"])
    if claim_type is None:
        errors.append("claim_id is not defined by qualitative_claim_types.yaml")
    else:
        if claim["polarity"] not in claim_type.polarities:
            errors.append("polarity is not allowed for claim_id")
        if claim["time_horizon"] not in claim_type.horizons:
            errors.append("time_horizon is not allowed for claim_id")
        if claim["strength"] not in claim_type.strengths:
            errors.append("strength is not allowed for claim_id")
    if not isinstance(claim["confidence"], (int, float)) or not 0 <= float(claim["confidence"]) <= 1:
        errors.append("confidence must be numeric in [0, 1]")
    if not isinstance(claim["start_char"], int) or not isinstance(claim["end_char"], int) or claim["start_char"] >= claim["end_char"]:
        errors.append("invalid evidence bounds")
    if claim["label_source"] != LABEL_SOURCE:
        errors.append("unexpected label_source")
    if claim["claim_extractor_version"] != CLAIM_EXTRACTOR_VERSION:
        errors.append("unexpected claim_extractor_version")
    if any(key in claim for key in FORBIDDEN_INPUT_OR_OUTPUT_KEYS):
        errors.append("forbidden raw text, route label, or outcome field present")
    return errors


def run_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
