"""Measure agreement between primary and blinded second-teacher labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

try:
    from .download_free_pilot import external_output
    from .validate_llm_teacher_labels import ALLOWED_MODELS
except ImportError:
    from download_free_pilot import external_output
    from validate_llm_teacher_labels import ALLOWED_MODELS


def audit(primary_path: Path, secondary_path: Path, output: Path) -> dict[str, Any]:
    primary_path, secondary_path, output = map(external_output, (primary_path, secondary_path, output))
    primary = {label["episode_id"]: label for label in json.loads(primary_path.read_text(encoding="utf-8"))}
    secondary = {label["episode_id"]: label for label in json.loads(secondary_path.read_text(encoding="utf-8"))}
    ids = sorted(set(primary) & set(secondary))
    if not ids:
        raise ValueError("teacher files have no overlapping episodes")
    jaccard = []
    exact_set = exact_order = abstain_agreement = 0
    for episode_id in ids:
        first, second = primary[episode_id], secondary[episode_id]
        first_set, second_set = set(first["selected_models"]), set(second["selected_models"])
        union = first_set | second_set
        jaccard.append(len(first_set & second_set) / len(union) if union else 1.0)
        exact_set += first_set == second_set
        exact_order += first["ordered_models"] == second["ordered_models"]
        abstain_agreement += bool(first["abstain"]) == bool(second["abstain"])
    kappas = {}
    for model in sorted(ALLOWED_MODELS):
        first_y = [int(model in primary[episode_id]["selected_models"]) for episode_id in ids]
        second_y = [int(model in secondary[episode_id]["selected_models"]) for episode_id in ids]
        if len(set(first_y)) == 1 and len(set(second_y)) == 1:
            kappas[model] = 1.0 if first_y == second_y else 0.0
        else:
            kappas[model] = float(cohen_kappa_score(first_y, second_y))
    report = {
        "overlap_episodes": len(ids),
        "mean_selected_set_jaccard": sum(jaccard) / len(jaccard),
        "exact_selected_set_rate": exact_set / len(ids),
        "exact_ordered_route_rate": exact_order / len(ids),
        "abstain_agreement_rate": abstain_agreement / len(ids),
        "per_model_cohen_kappa": kappas,
        "interpretation_boundary": "Agreement measures repeatability of the same LLM teacher family, not correctness or human ground truth.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(audit(args.primary, args.secondary, args.output))


if __name__ == "__main__":
    main()
