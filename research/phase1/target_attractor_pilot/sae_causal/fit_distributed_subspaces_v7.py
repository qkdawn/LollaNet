from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .contracts import AXES, read_jsonl
from .extract_distributed_residuals_v7 import TOKEN_WINDOWS


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PRIOR_SEEN_CASES = {"non_template": {"nv3-049", "nv3-062"}}


def _residualize_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_c = np.column_stack((np.ones(len(c_train)), c_train))
    test_c = np.column_stack((np.ones(len(c_test)), c_test))
    beta_x = np.linalg.lstsq(train_c, x_train, rcond=None)[0]
    beta_y = np.linalg.lstsq(train_c, y_train, rcond=None)[0]
    return (
        x_train - train_c @ beta_x,
        y_train - train_c @ beta_y,
        x_test - test_c @ beta_x,
        y_test - test_c @ beta_y,
    )


def _ridge_direction(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x @ x.T
    dual = np.linalg.solve(gram + np.eye(len(x), dtype=np.float64) * alpha, y)
    return (x.T @ dual).astype(np.float64)


def _safe_spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    value = float(spearmanr(y, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def grouped_oof_ridge(
    x: np.ndarray,
    y: np.ndarray,
    confounds: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    folds: int = 5,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    prediction = np.zeros(len(y), dtype=np.float64)
    target = np.zeros(len(y), dtype=np.float64)
    for train, test in splitter.split(x, y, groups):
        xtr, ytr, xte, yte = _residualize_fit(
            x[train], y[train], confounds[train], x[test], y[test], confounds[test]
        )
        scale = np.std(xtr, axis=0)
        scale[scale < 1e-6] = 1.0
        xtr_scaled, xte_scaled = xtr / scale, xte / scale
        direction = _ridge_direction(xtr_scaled, ytr, alpha)
        prediction[test] = xte_scaled @ direction
        target[test] = yte
    return {
        "spearman": _safe_spearman(target, prediction),
        "auroc": float(roc_auc_score(target > 0, prediction)),
        "sign_accuracy": float(np.mean(np.sign(target) == np.sign(prediction))),
        "predictions": prediction,
        "residualized_targets": target,
    }


def augment_bidirectional(
    x: np.ndarray, y: np.ndarray, confounds: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn oriented pair differences into a balanced bidirectional contrast."""
    return (
        np.concatenate((x, -x), axis=0),
        np.concatenate((np.ones(len(y)), -np.ones(len(y))), axis=0),
        np.concatenate((confounds, -confounds), axis=0),
        np.concatenate((groups, groups), axis=0),
    )


def _pair_rows(
    axis: str,
    discovery: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    record_index: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    other_axes = [name for name in AXES if name != axis]
    for pair in discovery["pairs"][axis]:
        if pair["split"] != "train":
            continue
        low_id, high_id = str(pair["low_record_id"]), str(pair["high_record_id"])
        if low_id not in record_index or high_id not in record_index:
            continue
        low, high = labels[low_id], labels[high_id]
        rows.append({
            **pair,
            "low_index": record_index[low_id],
            "high_index": record_index[high_id],
            "target_delta": float(high[axis]) - float(low[axis]),
            "confounds": [
                float(pair["response_token_delta"]),
                *[float(high[name]) - float(low[name]) for name in other_axes],
            ],
        })
    return rows


def _fit_final(
    x: np.ndarray, y: np.ndarray, confounds: np.ndarray, alpha: float
) -> tuple[np.ndarray, dict[str, Any]]:
    design = np.column_stack((np.ones(len(confounds)), confounds))
    beta_x = np.linalg.lstsq(design, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, y, rcond=None)[0]
    x_residual = x - design @ beta_x
    y_residual = y - design @ beta_y
    scale = np.std(x_residual, axis=0)
    scale[scale < 1e-6] = 1.0
    scaled_direction = _ridge_direction(x_residual / scale, y_residual, alpha)
    raw_direction = scaled_direction / scale
    norm = float(np.linalg.norm(raw_direction))
    if norm <= 1e-12:
        raise ValueError("distributed direction has zero norm")
    return (raw_direction / norm).astype(np.float32), {
        "confound_beta_y": beta_y.tolist(),
        "direction_norm_before_normalization": norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--extract-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    residuals = np.load(args.residuals, mmap_mode="r")
    mask = np.load(args.mask, mmap_mode="r")
    records = json.loads(args.records.read_text(encoding="utf-8"))
    extract_manifest = json.loads(args.extract_manifest.read_text(encoding="utf-8"))
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    fit_manifest = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    labels = {str(row["record_id"]): row for row in read_jsonl(args.labels)}
    record_index = {str(row["record_id"]): index for index, row in enumerate(records)}
    layers = [int(value) for value in extract_manifest["layers"]]
    if residuals.shape[:3] != mask.shape or residuals.shape[0] != len(records):
        raise ValueError("distributed residual artifact shape mismatch")

    results: dict[str, Any] = {}
    directions: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(2026080701)
    for axis in AXES:
        pairs = _pair_rows(axis, discovery, labels, record_index)
        fit_cases = set(fit_manifest["axes"][axis]["fit_case_ids"])
        holdout_cases = set(fit_manifest["axes"][axis]["internal_holdout_cases"])
        seen_cases = PRIOR_SEEN_CASES.get(axis, set())
        fit_pairs = [row for row in pairs if str(row["case_id"]) in fit_cases]
        holdout_pairs = [
            row for row in pairs
            if str(row["case_id"]) in holdout_cases and str(row["case_id"]) not in seen_cases
        ]
        candidates: list[dict[str, Any]] = []
        for layer_position, layer in enumerate(layers):
            for window_position, window in enumerate(TOKEN_WINDOWS):
                usable = [
                    row for row in fit_pairs
                    if mask[row["low_index"], layer_position, window_position]
                    and mask[row["high_index"], layer_position, window_position]
                ]
                if len(usable) < 10 or len({str(row["case_id"]) for row in usable}) < 5:
                    continue
                x = np.stack([
                    residuals[row["high_index"], layer_position, window_position].astype(np.float32)
                    - residuals[row["low_index"], layer_position, window_position].astype(np.float32)
                    for row in usable
                ])
                y = np.asarray([row["target_delta"] for row in usable], dtype=np.float64)
                confounds = np.asarray([row["confounds"] for row in usable], dtype=np.float64)
                groups = np.asarray([str(row["case_id"]) for row in usable])
                x, y, confounds, groups = augment_bidirectional(x, y, confounds, groups)
                for alpha in RIDGE_ALPHAS:
                    oof = grouped_oof_ridge(x, y, confounds, groups, alpha)
                    candidates.append({
                        "layer": layer,
                        "layer_position": layer_position,
                        "window": list(window),
                        "window_position": window_position,
                        "alpha": alpha,
                        "pairs": len(usable),
                        "cases": len(set(groups)),
                        "oof_spearman": oof["spearman"],
                        "oof_auroc": oof["auroc"],
                        "oof_sign_accuracy": oof["sign_accuracy"],
                    })
        if not candidates:
            raise ValueError(f"no distributed candidates for {axis}")
        selected = max(
            candidates,
            key=lambda row: (row["oof_auroc"], row["oof_sign_accuracy"], -row["alpha"]),
        )
        selected_fit = [
            row for row in fit_pairs
            if mask[row["low_index"], selected["layer_position"], selected["window_position"]]
            and mask[row["high_index"], selected["layer_position"], selected["window_position"]]
        ]
        x_fit = np.stack([
            residuals[row["high_index"], selected["layer_position"], selected["window_position"]].astype(np.float32)
            - residuals[row["low_index"], selected["layer_position"], selected["window_position"]].astype(np.float32)
            for row in selected_fit
        ])
        y_fit = np.asarray([row["target_delta"] for row in selected_fit], dtype=np.float64)
        c_fit = np.asarray([row["confounds"] for row in selected_fit], dtype=np.float64)
        fit_groups = np.asarray([str(row["case_id"]) for row in selected_fit])
        x_fit, y_fit, c_fit, _ = augment_bidirectional(
            x_fit, y_fit, c_fit, fit_groups
        )
        direction, final_meta = _fit_final(x_fit, y_fit, c_fit, float(selected["alpha"]))
        directions[axis] = direction

        usable_holdout = [
            row for row in holdout_pairs
            if mask[row["low_index"], selected["layer_position"], selected["window_position"]]
            and mask[row["high_index"], selected["layer_position"], selected["window_position"]]
        ]
        holdout_result: dict[str, Any] = {"pairs": len(usable_holdout), "cases": 0}
        if usable_holdout:
            x_holdout = np.stack([
                residuals[row["high_index"], selected["layer_position"], selected["window_position"]].astype(np.float32)
                - residuals[row["low_index"], selected["layer_position"], selected["window_position"]].astype(np.float32)
                for row in usable_holdout
            ])
            y_holdout = np.asarray([row["target_delta"] for row in usable_holdout])
            c_holdout = np.asarray([row["confounds"] for row in usable_holdout])
            holdout_groups = np.asarray([str(row["case_id"]) for row in usable_holdout])
            x_holdout, y_holdout, _, holdout_groups = augment_bidirectional(
                x_holdout, y_holdout, c_holdout, holdout_groups
            )
            prediction = x_holdout @ direction
            holdout_result = {
                "pairs": len(usable_holdout),
                "bidirectional_examples": len(y_holdout),
                "cases": len(set(holdout_groups)),
                "spearman": _safe_spearman(y_holdout, prediction),
                "auroc": float(roc_auc_score(y_holdout > 0, prediction)),
                "sign_accuracy": float(np.mean(np.sign(y_holdout) == np.sign(prediction))),
                "case_ids": sorted({str(row["case_id"]) for row in usable_holdout}),
            }
        random_direction = rng.normal(size=direction.shape).astype(np.float32)
        random_direction /= np.linalg.norm(random_direction)
        directions[f"{axis}__random"] = random_direction
        results[axis] = {
            "selected_on_fit_only": selected,
            "fit_direction": final_meta,
            "internal_holdout_report_only": holdout_result,
            "prior_seen_cases_excluded": sorted(seen_cases),
            "passes_fit_oof_gate": bool(
                selected["oof_auroc"] >= 0.65 and selected["oof_sign_accuracy"] >= 0.6
            ),
            "top_five_fit_candidates": sorted(
                candidates, key=lambda row: row["oof_auroc"], reverse=True
            )[:5],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "distributed_directions.npz", **directions)
    manifest = {
        "type": "distributed_residual_subspace_fit_v7",
        "status": "fit_complete",
        "selection_split": "train_fit_cases_grouped_oof",
        "development_accessed": False,
        "formal_evaluation_accessed": False,
        "confounds": ["response_token_delta", "other_axis_delta_1", "other_axis_delta_2"],
        "prompt_and_case_control": "same-prompt pair differencing plus case-grouped folds",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "axes": results,
        "evidence_boundary": (
            "Layer/window selection is train-fit-only association. Internal holdout is report-only; "
            "causal generation intervention remains required."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({axis: results[axis]["selected_on_fit_only"] for axis in AXES}, indent=2))


if __name__ == "__main__":
    main()
