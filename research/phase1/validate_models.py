"""Validate self-contained Phase 1 model assets without reading public model cards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def validate(root: Path = ROOT) -> list[str]:
    schema = load_yaml(root / "model_schema.yaml")
    models = [(path, load_yaml(path)) for path in sorted((root / "models").glob("*.yaml"))]
    errors: list[str] = []
    ids: set[str] = set()
    forbidden = {"source_card", "public_card", "public_model_version", "numeric_parameters"}

    for path, model in models:
        label = path.name
        missing = set(schema["required_fields"]) - set(model)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        if forbidden & set(model):
            errors.append(f"{label}: contains forbidden public or numeric fields")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.isascii() or not model_id.startswith("mm-p1-"):
            errors.append(f"{label}: id must be ASCII and start with mm-p1-")
        elif model_id in ids:
            errors.append(f"{label}: duplicate id {model_id}")
        else:
            ids.add(model_id)
        for field, allowed in (("domain", schema["domains"]), ("kind", schema["kinds"])):
            if model.get(field) not in allowed:
                errors.append(f"{label}: invalid {field}")
        roles = model.get("functional_roles")
        if not isinstance(roles, list) or not roles or not set(roles) <= set(schema["functional_roles"]):
            errors.append(f"{label}: invalid functional_roles")
        for field in ("definition", "mechanism"):
            if not isinstance(model.get(field), str) or not model[field].strip():
                errors.append(f"{label}: {field} must be non-empty text")
        for collection in ("key_variables", "input_fields", "output_fields"):
            values = model.get(collection)
            if not isinstance(values, list) or not values:
                errors.append(f"{label}: {collection} must be a non-empty list")
                continue
            for value in values:
                if not isinstance(value, dict) or not {"name", "type", "meaning"} <= set(value):
                    errors.append(f"{label}: malformed {collection} item")
                elif value["type"] not in schema["field_types"]:
                    errors.append(f"{label}: invalid field type {value['type']}")
        for field in (
            "positive_task_signals", "negative_task_signals", "preconditions",
            "failure_conditions", "prediction_targets", "action_affordances",
        ):
            if not isinstance(model.get(field), list) or not model[field]:
                errors.append(f"{label}: {field} must be a non-empty list")
        composition = model.get("composition")
        if not isinstance(composition, dict):
            errors.append(f"{label}: composition must be a mapping")
            continue
        missing_composition = set(schema["composition_fields"]) - set(composition)
        if missing_composition:
            errors.append(f"{label}: missing composition fields {sorted(missing_composition)}")
        elif not isinstance(composition["composition_patterns"], list) or not composition["composition_patterns"]:
            errors.append(f"{label}: requires at least one composition pattern")

    for path, model in models:
        composition = model.get("composition", {})
        references: list[str] = []
        for field in ("prerequisites", "successors", "compatible_with", "conflicts_with"):
            references.extend(composition.get(field, []))
        for pattern in composition.get("composition_patterns", []):
            if not isinstance(pattern, dict) or not isinstance(pattern.get("route"), list):
                errors.append(f"{path.name}: malformed composition pattern")
                continue
            route = pattern["route"]
            if len(route) < 2 or model.get("id") not in route:
                errors.append(f"{path.name}: composition route must include this model and another model")
            references.extend(route)
        unknown = set(references) - ids
        if unknown:
            errors.append(f"{path.name}: unknown model references {sorted(unknown)}")
    if len(models) != 20:
        errors.append(f"expected 20 model assets, found {len(models)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        print("Phase 1 model asset validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("Phase 1 model assets are valid (20 models).")


if __name__ == "__main__":
    main()
