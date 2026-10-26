from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .analyze_temporal_v3 import _feature_trajectory
from .contracts import AXES


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("subspace matrix must have at least one column")
    basis, triangular = np.linalg.qr(values)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.sum(diagonal > 1e-8))
    if rank != values.shape[1]:
        raise ValueError("subspace directions are rank deficient")
    return basis[:, :rank].astype(np.float32)


def project_delta(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float32)
    space = np.asarray(basis, dtype=np.float32)
    return (values @ space @ space.T).astype(np.float32)


def _load_temporal_codes(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        return archive["record_ids"].tolist(), archive["indices"].copy(), archive["values"].copy()


def _record_feature_matrix(
    indices: np.ndarray, values: np.ndarray, features: list[int]
) -> np.ndarray:
    return np.stack([
        _feature_trajectory(indices, values, feature).mean(axis=1)
        for feature in features
    ], axis=1).astype(np.float64)


def _pair_differences(
    matrix: np.ndarray,
    record_index: dict[str, int],
    pairs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.asarray([record_index[str(pair["low_record_id"])] for pair in pairs])
    high = np.asarray([record_index[str(pair["high_record_id"])] for pair in pairs])
    return (
        matrix[high] - matrix[low],
        np.asarray([float(pair["target_delta"]) for pair in pairs], dtype=np.float64),
        np.asarray([str(pair["case_id"]) for pair in pairs]),
    )


def fit_sparse_membership(
    differences: np.ndarray,
    target_delta: np.ndarray,
    cases: np.ndarray,
    *,
    minimum_rank: int = 2,
    maximum_rank: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    design = np.vstack((-0.5 * differences, 0.5 * differences))
    outcome = np.concatenate((-0.5 * target_delta, 0.5 * target_delta))
    groups = np.concatenate((cases, cases))
    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(design)
    folds = min(5, len(np.unique(groups)))
    split = list(GroupKFold(folds).split(scaled, outcome, groups))
    model = ElasticNetCV(
        l1_ratio=[0.5, 0.8, 1.0],
        alphas=np.logspace(-5, -0.2, 20),
        fit_intercept=False,
        cv=split,
        max_iter=20000,
        random_state=20260727,
    ).fit(scaled, outcome)
    coefficients = model.coef_ / np.maximum(scaler.scale_, 1e-8)
    order = np.argsort(np.abs(coefficients))[::-1]
    nonzero = order[np.abs(coefficients[order]) > 1e-8]
    rank = min(maximum_rank, len(coefficients), max(minimum_rank, len(nonzero)))
    selected = order[:rank]
    return selected.astype(np.int64), {
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "coefficients": coefficients.tolist(),
        "selected_columns": selected.tolist(),
        "selected_rank": int(rank),
        "minimum_rank_enforced": len(nonzero) < minimum_rank,
    }


def _activity_matched_controls(
    indices: np.ndarray,
    targets: list[int],
    excluded: set[int],
) -> list[int]:
    valid = indices.reshape(-1)
    valid = valid[valid >= 0]
    counts = np.bincount(valid, minlength=65536)
    chosen: list[int] = []
    for target in targets:
        eligible = [
            int(value) for value in np.flatnonzero(counts > 0)
            if int(value) not in excluded and int(value) not in chosen
        ]
        control = min(eligible, key=lambda value: (
            abs(np.log1p(float(counts[value])) - np.log1p(float(counts[target]))), value
        ))
        chosen.append(control)
    return chosen


def _load_mean_residuals(
    record_ids: list[str], shard_directory: Path, token_limit: int = 16
) -> np.ndarray:
    rows = []
    for record_id in record_ids:
        values = np.load(shard_directory / f"{record_id}.npy")[:token_limit]
        rows.append(values.mean(axis=0))
    return np.stack(rows).astype(np.float32)


def fit_probe_mediated_das(
    differences: np.ndarray,
    target_delta: np.ndarray,
    rank: int,
    *,
    seed: int,
    probe_steps: int = 600,
    das_steps: int = 800,
) -> tuple[np.ndarray, dict[str, float]]:
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta = torch.as_tensor(differences, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_delta, dtype=torch.float32, device=device)
    design = torch.cat((-0.5 * delta, 0.5 * delta), dim=0)
    outcome = torch.cat((-0.5 * target, 0.5 * target), dim=0)[:, None]
    probe = torch.nn.Sequential(
        torch.nn.LayerNorm(design.shape[1]),
        torch.nn.Linear(design.shape[1], 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(probe_steps):
        prediction = probe(design)
        loss = torch.nn.functional.mse_loss(prediction, outcome)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        probe_mse = float(torch.nn.functional.mse_loss(probe(design), outcome).cpu())

    _left, _singular, right = torch.linalg.svd(delta, full_matrices=False)
    initialization = right[:rank].T.contiguous()
    raw_basis = torch.nn.Parameter(initialization + 0.01 * torch.randn_like(initialization))
    optimizer = torch.optim.Adam([raw_basis], lr=5e-3)
    low, high = -0.5 * delta, 0.5 * delta
    low_target, high_target = -0.5 * target[:, None], 0.5 * target[:, None]
    initial_loss = None
    for step in range(das_steps):
        basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        projection_delta = (high - low) @ basis @ basis.T
        swapped_low = low + projection_delta
        swapped_high = high - projection_delta
        loss = (
            torch.nn.functional.mse_loss(probe(swapped_low), high_target)
            + torch.nn.functional.mse_loss(probe(swapped_high), low_target)
        ) / 2
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        projection_delta = (high - low) @ basis @ basis.T
        final_loss = float((
            torch.nn.functional.mse_loss(probe(low + projection_delta), high_target)
            + torch.nn.functional.mse_loss(probe(high - projection_delta), low_target)
        ).cpu() / 2)
    return basis.cpu().numpy().astype(np.float32), {
        "probe_train_mse": probe_mse,
        "interchange_initial_mse": float(initial_loss),
        "interchange_final_mse": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--temporal-analysis", type=Path, required=True)
    parser.add_argument("--temporal-codes", type=Path, required=True)
    parser.add_argument("--residual-shards", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal_analysis.read_text(encoding="utf-8"))
    record_ids, indices, values = _load_temporal_codes(args.temporal_codes)
    record_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    mean_residuals = _load_mean_residuals(record_ids, args.residual_shards)
    raw_sae = torch.load(args.sae, map_location="cpu", weights_only=True)
    decoder = raw_sae["W_dec"].float().numpy()
    if decoder.shape[0] != mean_residuals.shape[1]:
        decoder = decoder.T
    del raw_sae
    gc.collect()

    all_candidates = {
        int(feature)
        for axis in AXES
        for feature in temporal["axes"][axis]["early_available_candidates"]
    }
    bases: dict[str, np.ndarray] = {}
    manifest_axes: dict[str, Any] = {}
    for axis_index, axis in enumerate(AXES):
        features = [int(value) for value in temporal["axes"][axis]["early_available_candidates"]]
        feature_matrix = _record_feature_matrix(indices, values, features)
        train_pairs = [pair for pair in discovery["pairs"][axis] if pair["split"] == "train"]
        feature_delta, target_delta, cases = _pair_differences(
            feature_matrix, record_index, train_pairs
        )
        selected_columns, fit = fit_sparse_membership(feature_delta, target_delta, cases)
        selected_features = [features[int(column)] for column in selected_columns]
        sae_basis = orthonormalize(decoder[:, selected_features])
        random_features = _activity_matched_controls(
            indices, selected_features, all_candidates
        )
        random_sae_basis = orthonormalize(decoder[:, random_features])

        residual_delta, _, _ = _pair_differences(mean_residuals, record_index, train_pairs)
        das_basis, das_fit = fit_probe_mediated_das(
            residual_delta,
            target_delta,
            sae_basis.shape[1],
            seed=2026072700 + axis_index,
        )
        rng = np.random.default_rng(2026072800 + axis_index)
        random_das_basis = orthonormalize(
            rng.standard_normal((mean_residuals.shape[1], sae_basis.shape[1]))
        )
        bases[f"{axis}__sae"] = sae_basis
        bases[f"{axis}__sae_random"] = random_sae_basis
        bases[f"{axis}__das"] = das_basis
        bases[f"{axis}__das_random"] = random_das_basis
        manifest_axes[axis] = {
            "candidate_features": features,
            "selected_features": selected_features,
            "random_features": random_features,
            "rank": int(sae_basis.shape[1]),
            "elastic_net": fit,
            "das_fit": das_fit,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "subspace_bases.npz", **bases)
    (args.output / "fit_manifest.json").write_text(json.dumps({
        "type": "sae_vs_probe_mediated_das_subspaces_v4",
        "status": "ready_for_development_interchange",
        "selection_split": "train",
        "development_accessed_during_fit": False,
        "axes": manifest_axes,
        "evidence_boundary": (
            "Probe-mediated DAS fit is a candidate distributed subspace, not behavioral evidence. "
            "Only generation interchange can test causal behavioral effects."
        ),
    }, indent=2), encoding="utf-8")
    print(json.dumps(manifest_axes, indent=2))


if __name__ == "__main__":
    main()
