from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from .contracts import AXES


@dataclass
class ResidualizationPipeline:
    categorical: list[str]
    numeric: list[str]
    categories: dict[str, list[str]]
    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def design_matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        one_hot_columns = []
        for column in self.categorical:
            values = np.asarray([str(row[column]) for row in rows])
            one_hot_columns.extend((values == category).astype(np.float64) for category in self.categories[column])
        numeric = np.asarray([[float(row[column]) for column in self.numeric] for row in rows], dtype=np.float64)
        numeric = (numeric - self.numeric_mean) / self.numeric_scale
        columns = one_hot_columns + [numeric[:, index] for index in range(numeric.shape[1])]
        return np.column_stack(columns)

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.design_matrix(rows) @ self.coefficients + self.intercept


@dataclass
class ResidualizedLabels:
    record_ids: list[str]
    split: np.ndarray
    case_ids: np.ndarray
    raw: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]
    pipelines: dict[str, ResidualizationPipeline]


def visible_markers(text: str) -> dict[str, float]:
    lines = text.splitlines()
    return {
        "response_characters": float(len(text)),
        "heading_count": float(sum(bool(re.match(r"^\s*#{1,6}\s", line)) for line in lines)),
        "bullet_count": float(sum(bool(re.match(r"^\s*[-*+]\s", line)) for line in lines)),
        "numbered_count": float(sum(bool(re.match(r"^\s*\d+[.)]\s", line)) for line in lines)),
        "paragraph_count": float(sum(bool(line.strip()) and (i == 0 or not lines[i - 1].strip()) for i, line in enumerate(lines))),
    }


def _design_rows(trajectories: list[dict[str, Any]], labels: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): row for row in trajectories}
    output = []
    for label in labels:
        trajectory = by_id[str(label["record_id"])]
        markers = visible_markers(str(trajectory["response"]))
        output.append({
            "case_id": str(label["case_id"]), "domain": str(label["domain"]),
            "response_tokens": float(trajectory["response_tokens"]),
            "generation_finished": float(bool(trajectory["generation_finished"])), **markers,
            **{f"other_{axis}": float(label[axis]) for axis in AXES if axis != target},
        })
    return output


def _fit_residualizer(
    rows: list[dict[str, Any]], train: np.ndarray, values: np.ndarray, alpha: float
) -> ResidualizationPipeline:
    categorical = ["case_id", "domain"]
    numeric = [column for column in rows[0] if column not in categorical]
    categories = {
        column: sorted({str(row[column]) for row, is_train in zip(rows, train) if is_train})
        for column in categorical
    }
    train_numeric = np.asarray(
        [[float(row[column]) for column in numeric] for row, is_train in zip(rows, train) if is_train],
        dtype=np.float64,
    )
    numeric_mean = train_numeric.mean(axis=0)
    numeric_scale = train_numeric.std(axis=0)
    numeric_scale[numeric_scale == 0.0] = 1.0
    unfitted = ResidualizationPipeline(
        categorical, numeric, categories, numeric_mean, numeric_scale,
        np.empty(sum(len(items) for items in categories.values()) + len(numeric)), 0.0,
    )
    design = unfitted.design_matrix(rows)[train]
    target = values[train]
    design_mean = design.mean(axis=0)
    target_mean = float(target.mean())
    centered_design = design - design_mean
    centered_target = target - target_mean
    gram = centered_design.T @ centered_design
    if alpha > 0.0:
        coefficients = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0]), centered_design.T @ centered_target
        )
    else:
        coefficients = np.linalg.lstsq(centered_design, centered_target, rcond=None)[0]
    unfitted.coefficients = coefficients
    unfitted.intercept = target_mean - float(design_mean @ coefficients)
    return unfitted


def residualize_axes(trajectories: list[dict[str, Any]], labels: list[dict[str, Any]], alpha: float = 10.0) -> ResidualizedLabels:
    if [str(row["record_id"]) for row in trajectories] != [str(row["record_id"]) for row in labels]:
        by_id = {str(row["record_id"]): row for row in trajectories}
        trajectories = [by_id[str(row["record_id"])] for row in labels]
    split = np.asarray([str(row["split"]) for row in labels])
    train = split == "train"
    if not train.any() or not (split == "development").any():
        raise ValueError("residualization requires both train and development rows")
    residuals: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    pipelines: dict[str, ResidualizationPipeline] = {}
    for axis in AXES:
        rows = _design_rows(trajectories, labels, axis)
        values = np.asarray([float(row[axis]) for row in labels], dtype=np.float64)
        pipeline = _fit_residualizer(rows, train, values, alpha)
        residuals[axis] = values - pipeline.predict(rows)
        raw[axis] = values
        pipelines[axis] = pipeline
    return ResidualizedLabels(
        record_ids=[str(row["record_id"]) for row in labels], split=split,
        case_ids=np.asarray([str(row["case_id"]) for row in labels]), raw=raw,
        residual=residuals, pipelines=pipelines,
    )
