from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers

from .contracts import AXES, read_jsonl
from .intervention_smoke_v2 import _atomic_jsonl
from .resources import require_commit_headroom
from .subspace_interchange_smoke_v4 import (
    METHODS,
    _generate,
    norm_match,
    subspace_swap_update,
)


ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
NONZERO_ALPHAS = tuple(alpha for alpha in ALPHAS if alpha != 0.0)
DIRECTIONS = (
    ("high-to-low", "high", "low"),
    ("low-to-high", "low", "high"),
)


def _alpha_label(alpha: float) -> str:
    return ("m" if alpha < 0 else "p") + str(abs(alpha)).replace(".", "p")


def select_train_pairs(
    discovery: dict[str, Any], axis: str, count: int
) -> list[dict[str, Any]]:
    pairs = sorted(
        (pair for pair in discovery["pairs"][axis] if pair.get("split") == "train"),
        key=lambda pair: str(pair["pair_id"]),
    )
    if len(pairs) < count:
        raise ValueError(f"axis {axis} has only {len(pairs)} train pairs")
    selected = pairs[:count]
    if any(pair.get("split") != "train" for pair in selected):
        raise ValueError("non-train pair selected")
    return selected


def build_plan(discovery: dict[str, Any], pairs_per_axis: int = 2) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for axis in AXES:
        for pair in select_train_pairs(discovery, axis, pairs_per_axis):
            pair_id = str(pair["pair_id"])
            for direction, source_class, target_class in DIRECTIONS:
                plan.append({
                    "record_id": f"signed-v4__{pair_id}__baseline__{direction}",
                    "pair_id": pair_id,
                    "axis": axis,
                    "condition": "baseline",
                    "direction": direction,
                    "source_class": source_class,
                    "target_class": target_class,
                    "alpha": 0.0,
                })
                for alpha in NONZERO_ALPHAS:
                    for method in METHODS:
                        plan.append({
                            "record_id": (
                                f"signed-v4__{pair_id}__{method}__{direction}"
                                f"__alpha-{_alpha_label(alpha)}"
                            ),
                            "pair_id": pair_id,
                            "axis": axis,
                            "condition": "signed_dose",
                            "direction": direction,
                            "source_class": source_class,
                            "target_class": target_class,
                            "method": method,
                            "alpha": alpha,
                        })
    expected = len(AXES) * pairs_per_axis * (
        len(DIRECTIONS) + len(DIRECTIONS) * len(NONZERO_ALPHAS) * len(METHODS)
    )
    ids = [row["record_id"] for row in plan]
    if len(plan) != expected or len(ids) != len(set(ids)):
        raise ValueError("signed-dose plan count or ID uniqueness failure")
    return plan


def audit_inputs(
    trajectories: list[dict[str, Any]],
    discovery: dict[str, Any],
    fit: dict[str, Any],
    bases: dict[str, np.ndarray],
    residual_shards: Path,
    pairs_per_axis: int,
) -> dict[str, Any]:
    if fit.get("selection_split") != "train" or fit.get("development_accessed_during_fit") is not False:
        raise ValueError("subspace fit is not sealed train-only")
    trajectory_by_id = {str(row["record_id"]): row for row in trajectories}
    plan = build_plan(discovery, pairs_per_axis)
    selected_ids: set[str] = set()
    selected_pairs: dict[str, list[str]] = {}
    for axis in AXES:
        selected_pairs[axis] = []
        for pair in select_train_pairs(discovery, axis, pairs_per_axis):
            selected_pairs[axis].append(str(pair["pair_id"]))
            for key in ("low_record_id", "high_record_id"):
                record_id = str(pair[key])
                selected_ids.add(record_id)
                row = trajectory_by_id.get(record_id)
                if row is None or row.get("split") != "train":
                    raise ValueError(f"selected record {record_id} is missing or not train")
                if not (residual_shards / f"{record_id}.npy").is_file():
                    raise ValueError(f"missing residual shard for {record_id}")
        ranks = {np.asarray(bases[f"{axis}__{method}"]).shape[1] for method in METHODS}
        if ranks != {int(fit["axes"][axis]["rank"])}:
            raise ValueError(f"axis {axis} subspaces are not rank matched")
    return {
        "status": "dry_run_passed",
        "selection_split": "train",
        "development_pair_signs_accessed": False,
        "pairs_per_axis": pairs_per_axis,
        "selected_pairs": selected_pairs,
        "selected_record_count": len(selected_ids),
        "alphas": list(ALPHAS),
        "methods": list(METHODS),
        "directions": [direction for direction, _, _ in DIRECTIONS],
        "expected_records": len(plan),
        "baseline_records": len(AXES) * pairs_per_axis * len(DIRECTIONS),
        "intervention_records": len(plan) - len(AXES) * pairs_per_axis * len(DIRECTIONS),
        "evidence_boundary": (
            "Train-only recoverable discovery smoke. Probe-mediated DAS is not formal DAS. "
            "No formal evaluation or development pair sign selection is permitted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--residual-shards", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pairs-per-axis", type=int, default=2)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trajectories = read_jsonl(args.trajectories)
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    with np.load(args.bases) as archive:
        bases = {key: archive[key].copy() for key in archive.files}
    audit = audit_inputs(
        trajectories, discovery, fit, bases, args.residual_shards, args.pairs_per_axis
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dry_run_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return
    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    layer = _layers(model)[args.layer]
    trajectory_by_id = {str(row["record_id"]): row for row in trajectories}
    plan = build_plan(discovery, args.pairs_per_axis)
    plan_by_id = {row["record_id"]: row for row in plan}
    output_path = args.output / "signed_dose_trajectories.jsonl"
    existing = read_jsonl(output_path) if output_path.is_file() else []
    completed = {str(row["record_id"]) for row in existing}
    results = list(existing)

    for axis_index, axis in enumerate(AXES):
        for pair_index, pair in enumerate(select_train_pairs(discovery, axis, args.pairs_per_axis)):
            seed = 2026080200 + axis_index * 100 + pair_index
            low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
            residuals = {
                record_id: np.load(args.residual_shards / f"{record_id}.npy")[:16]
                for record_id in (low_id, high_id)
            }
            for direction, source_class, target_class in DIRECTIONS:
                source_id = high_id if source_class == "high" else low_id
                target_id = low_id if target_class == "low" else high_id
                target_row = trajectory_by_id[target_id]
                baseline_id = f"signed-v4__{pair['pair_id']}__baseline__{direction}"
                if baseline_id not in completed:
                    text, state = _generate(
                        model, tokenizer, layer, str(target_row["prompt"]),
                        str(target_row["response"]), seed, args.maximum_new_tokens, None,
                    )
                    results.append({
                        **plan_by_id[baseline_id], "case_id": pair["case_id"],
                        "source_record_id": source_id, "target_record_id": target_id,
                        "seed": seed, "prompt": target_row["prompt"], "response": text, **state,
                    })
                    completed.add(baseline_id)
                    _atomic_jsonl(output_path, results)
                raw = {
                    method: subspace_swap_update(
                        residuals[source_id], residuals[target_id], bases[f"{axis}__{method}"]
                    )
                    for method in METHODS
                }
                matched = dict(raw)
                scales = {method: 1.0 for method in METHODS}
                matched["sae_random"], scales["sae_random"] = norm_match(raw["sae_random"], raw["sae"])
                matched["das_random"], scales["das_random"] = norm_match(raw["das_random"], raw["das"])
                for alpha in NONZERO_ALPHAS:
                    for method in METHODS:
                        record_id = (
                            f"signed-v4__{pair['pair_id']}__{method}__{direction}"
                            f"__alpha-{_alpha_label(alpha)}"
                        )
                        if record_id in completed:
                            continue
                        update = (matched[method] * np.float32(alpha)).astype(np.float32)
                        text, state = _generate(
                            model, tokenizer, layer, str(target_row["prompt"]),
                            str(target_row["response"]), seed, args.maximum_new_tokens, update,
                        )
                        results.append({
                            **plan_by_id[record_id], "case_id": pair["case_id"],
                            "rank": int(bases[f"{axis}__{method}"].shape[1]),
                            "source_record_id": source_id, "target_record_id": target_id,
                            "norm_match_scale": scales[method], "seed": seed,
                            "prompt": target_row["prompt"], "response": text, **state,
                        })
                        completed.add(record_id)
                        _atomic_jsonl(output_path, results)
                        print(f"signed-dose {len(results)}/{len(plan)}", flush=True)

    manifest = {
        **audit,
        "type": "sae_vs_probe_mediated_das_signed_dose_train_smoke_v4",
        "status": "ready_for_blind_scoring" if len(results) == len(plan) else "in_progress",
        "records": len(results),
        "expected_records": len(plan),
        "model_assisted_scoring": "pending",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
