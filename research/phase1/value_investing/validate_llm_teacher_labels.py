"""Validate and combine LLM teacher-label batches outside the repository."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .download_free_pilot import external_output
except ImportError:
    from download_free_pilot import external_output


MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"
MODEL_ASSETS = {
    payload["id"]: payload
    for path in MODEL_ROOT.glob("*.yaml")
    if (payload := yaml.safe_load(path.read_text(encoding="utf-8")))
}
ALLOWED_MODELS = set(MODEL_ASSETS)
CROSS_DOMAIN_MODELS = {model_id for model_id, asset in MODEL_ASSETS.items() if asset["domain"] in {"physics", "chemistry"}}
REQUIRED = {"episode_id", "teacher_version", "selected_models", "ordered_models", "confidence", "evidence_fields", "uncertainty_reasons", "abstain", "model_rationales", "cross_domain_mappings"}


def validate_label(label: dict[str, Any], expected_episode: str, packet: dict[str, Any] | None = None) -> list[str]:
    errors = []
    if label.get("episode_id") != expected_episode:
        errors.append("episode_id does not match packet")
    if not REQUIRED <= set(label):
        errors.append(f"missing fields {sorted(REQUIRED - set(label))}")
        return errors
    selected, ordered = label["selected_models"], label["ordered_models"]
    if not label["abstain"] and not 3 <= len(selected) <= 7:
        errors.append("non-abstained selection must contain 3-7 models")
    if len(selected) != len(set(selected)) or set(selected) - ALLOWED_MODELS:
        errors.append("selected_models contains duplicates or unknown models")
    if set(ordered) != set(selected) or len(ordered) != len(selected):
        errors.append("ordered_models must be a permutation of selected_models")
    if not isinstance(label["confidence"], (int, float)) or not 0 <= label["confidence"] <= 1:
        errors.append("confidence must be between 0 and 1")
    if not isinstance(label["evidence_fields"], list) or not isinstance(label["uncertainty_reasons"], list):
        errors.append("evidence_fields and uncertainty_reasons must be lists")
    elif packet is not None:
        visible_features = set(packet["financial_features"]) | set(packet["market_state"])
        if set(label["evidence_fields"]) - visible_features:
            errors.append("evidence_fields contains a non-visible feature")
    rationales = label["model_rationales"]
    if not isinstance(rationales, dict) or set(rationales) != set(selected):
        errors.append("model_rationales must contain exactly the selected models")
    elif packet is not None:
        visible_features = set(packet["financial_features"]) | set(packet["market_state"])
        for model_id, rationale in rationales.items():
            if not isinstance(rationale, str) or not any(field in rationale for field in visible_features):
                errors.append(f"{model_id} rationale must cite a packet-specific visible feature name")
    mappings = label["cross_domain_mappings"]
    if not isinstance(mappings, list):
        errors.append("cross_domain_mappings must be a list")
    else:
        mapped_ids = {mapping.get("model_id") for mapping in mappings if isinstance(mapping, dict)}
        required_mappings = set(selected) & CROSS_DOMAIN_MODELS
        if mapped_ids != required_mappings or len(mappings) != len(required_mappings):
            errors.append("every selected physics/chemistry model requires exactly one cross-domain mapping")
        for mapping in mappings:
            if not isinstance(mapping, dict) or not {"model_id", "source_fields", "mechanism"} <= set(mapping) or not mapping.get("source_fields"):
                errors.append("cross-domain mappings require model_id, non-empty source_fields, and mechanism")
                break
            if packet is not None:
                visible = packet["financial_features"] | packet["market_state"]
                if any(field not in visible or visible[field] is None for field in mapping["source_fields"]):
                    errors.append("cross-domain mapping source_fields must be visible and non-null")
                    break
    return errors


def combine(sample_root: Path, labels_root: Path, output: Path) -> dict[str, Any]:
    sample_root, labels_root, output = map(external_output, (sample_root, labels_root, output))
    errors, combined = [], []
    batch_paths = sorted((sample_root / "batches").glob("batch_*.json"))
    for batch_path in batch_paths:
        packets = json.loads(batch_path.read_text(encoding="utf-8"))
        label_path = labels_root / batch_path.name.replace("batch_", "labels_")
        if not label_path.exists():
            errors.append(f"missing {label_path.name}")
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        if len(labels) != len(packets):
            errors.append(f"{label_path.name}: label count does not match packet count")
            continue
        for packet, label in zip(packets, labels):
            label_errors = validate_label(label, packet["episode_id"], packet)
            for error in label_errors:
                errors.append(f"{label_path.name}/{packet['episode_id']}: {error}")
            visible = packet["financial_features"] | packet["market_state"]
            label["grounded_evidence_values"] = {field: visible[field] for field in label.get("evidence_fields", []) if field in visible}
            for mapping in label.get("cross_domain_mappings", []):
                mapping["source_values"] = {field: visible[field] for field in mapping.get("source_fields", []) if field in visible}
            combined.append(label)
    if errors:
        raise ValueError("\n".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(label, sort_keys=True) + "\n" for label in combined), encoding="utf-8")
    model_frequency = Counter(model for label in combined for model in label["selected_models"])
    cross_domain_frequency = Counter(mapping["model_id"] for label in combined for mapping in label["cross_domain_mappings"])
    route_frequency = Counter(" -> ".join(label["ordered_models"]) for label in combined if not label["abstain"])
    report = {
        "labels": len(combined),
        "abstained": sum(bool(label["abstain"]) for label in combined),
        "mean_confidence": sum(float(label["confidence"]) for label in combined) / len(combined) if combined else None,
        "model_frequency": dict(model_frequency.most_common()),
        "cross_domain_frequency": dict(cross_domain_frequency.most_common()),
        "unique_ordered_routes": len(route_frequency),
        "most_common_routes": dict(route_frequency.most_common(10)),
        "selection_size_frequency": dict(sorted(Counter(len(label["selected_models"]) for label in combined).items())),
        "unselected_candidate_models": sorted(ALLOWED_MODELS - set(model_frequency)),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(combine(args.sample_root, args.labels_root, args.output))


if __name__ == "__main__":
    main()
