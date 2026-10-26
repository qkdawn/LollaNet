"""Recoverable training/evaluation entrypoint for microconcept encoder v1."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .data import (
    CONCEPT_IDS, FEATURE_NAMES, EncoderData, Example, SplitBundle, TrainOnlyNormalizer,
    atomic_write_json, atomic_write_jsonl, build_encoder_data, encode_claims, encode_text,
    make_splits, serializable_split_rows, target_arrays,
)
from .evaluation import (
    IGNORE, Temperatures, evaluate_predictions, multitask_loss, prediction_rows,
    rule_projection_rows, select_temperatures, task_pair_metrics,
)
from .model import DualHeadMicroconceptEncoder, ModelVariant, VARIANTS, select_node_logits

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config_v1.yaml"


class EncodedRows(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[Example], normalizer: TrainOnlyNormalizer, text_dim: int, shuffle_qg: bool, seed: int) -> None:
        self.examples = rows
        financial, _ = normalizer.transform(rows)
        claims = encode_claims(rows)
        text = encode_text(rows, text_dim, shuffle_qg, seed)
        targets = target_arrays(rows)
        self.tensors: dict[str, torch.Tensor] = {
            "financial": torch.from_numpy(financial), "claim_id": torch.from_numpy(claims[0]),
            "polarity": torch.from_numpy(claims[1]), "horizon": torch.from_numpy(claims[2]),
            "strength": torch.from_numpy(claims[3]), "confidence": torch.from_numpy(claims[4]),
            "text": torch.from_numpy(text), **{key: torch.from_numpy(value) for key, value in targets.items()},
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**{key: value[index] for key, value in self.tensors.items()}, "example": self.examples[index]}


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"examples": [row["example"] for row in rows]}
    for key in rows[0]:
        if key != "example":
            result[key] = torch.stack([row[key] for row in rows])
    return result


def _loader(dataset: EncodedRows, batch_size: int, shuffle: bool = False) -> DataLoader[dict[str, Any]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid config {path}")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pretrain_rows(rows: Iterable[Example]) -> list[Example]:
    return [row for row in rows if row.effective_tier in {"Weak", "Silver"}]


def _gold_rows(rows: Iterable[Example]) -> list[Example]:
    return [row for row in rows if row.effective_tier == "Gold"]


def _run_epoch(model: DualHeadMicroconceptEncoder, loader: DataLoader[dict[str, Any]], optimizer: torch.optim.Optimizer | None, device: torch.device, weights: dict[str, float], include_task: bool = True) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items() if isinstance(value, torch.Tensor)}
        outputs = model(batch)
        loss, _ = multitask_loss(outputs, batch, weights, include_task=include_task)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _fit_variant(variant: ModelVariant, train_rows: list[Example], dev_rows: list[Example], normalizer: TrainOnlyNormalizer, config: dict[str, Any], device: torch.device, seed: int, output_dir: Path) -> tuple[DualHeadMicroconceptEncoder, dict[str, Any], Temperatures]:
    training = config["training"]
    text_dim = int(config["architecture"]["text_dim"])
    hidden_dim = int(config["architecture"]["hidden_dim"])
    batch_size = int(training["batch_size"])
    pretrain = _pretrain_rows(train_rows)
    gold = _gold_rows(train_rows)
    if not pretrain:
        raise RuntimeError("no Weak/Silver pretraining labels in temporal training split")
    trials: list[tuple[float, dict[str, Any], dict[str, torch.Tensor], TrainOnlyNormalizer]] = []
    for profile_index, profile in enumerate(training["loss_profiles"]):
        _set_seed(seed + profile_index)
        train_set = EncodedRows(pretrain, normalizer, text_dim, variant.shuffle_qg_train, seed + profile_index)
        dev_set = EncodedRows(dev_rows, normalizer, text_dim, False, seed)
        model = DualHeadMicroconceptEncoder(train_set.tensors["financial"].shape[1], text_dim, hidden_dim, variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        weights = {key: float(profile[key]) for key in ("intensity", "direction", "applicability", "relevance")}
        history: list[dict[str, float]] = []
        best_state: dict[str, torch.Tensor] | None = None; best_dev = float("inf")
        for epoch in range(int(training["epochs"])):
            tr = _run_epoch(model, _loader(train_set, batch_size, shuffle=True), optimizer, device, weights, include_task=not variant.evidence_only)
            dv = _run_epoch(model, _loader(dev_set, batch_size), None, device, weights, include_task=not variant.evidence_only)
            history.append({"epoch": epoch + 1, "train_loss": tr, "development_loss": dv})
            if dv < best_dev:
                best_dev = dv; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert best_state is not None
        trials.append((best_dev, {"profile": profile, "history": history, "gold_finetune_rows": len(gold)}, best_state, normalizer))
    best_dev, details, state, _ = min(trials, key=lambda item: item[0])
    # Rebuild selected architecture and then carry out Gold fine-tuning only when
    # genuine Gold labels exist.  The supplied pilot normally takes the skip path.
    selected_profile = details["profile"]
    _set_seed(seed)
    sample = EncodedRows(pretrain[:1], normalizer, text_dim, False, seed)
    model = DualHeadMicroconceptEncoder(sample.tensors["financial"].shape[1], text_dim, hidden_dim, variant).to(device)
    model.load_state_dict(state)
    fine_tune_history: list[dict[str, float]] = []
    if gold:
        gold_set = EncodedRows(gold, normalizer, text_dim, variant.shuffle_qg_train, seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]) * 0.25, weight_decay=float(training["weight_decay"]))
        weights = {key: float(selected_profile[key]) for key in ("intensity", "direction", "applicability", "relevance")}
        for epoch in range(max(3, int(training["epochs"]) // 4)):
            fine_tune_history.append({"epoch": epoch + 1, "gold_train_loss": _run_epoch(model, _loader(gold_set, batch_size, shuffle=True), optimizer, device, weights, include_task=not variant.evidence_only)})
    # Temperature fitting uses development labels only and follows final selected model.
    dev_set = EncodedRows(dev_rows, normalizer, text_dim, False, seed)
    dev_predictions = prediction_rows(model, _loader(dev_set, batch_size), device, dev_rows)
    temps = select_temperatures(dev_predictions, config["calibration"]["temperature_grid"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "variant": asdict(variant), "normalizer": normalizer.to_dict(), "selected_profile": selected_profile, "temperatures": temps.to_dict()}, output_dir / f"{variant.name}.pt")
    return model, {
        "variant": asdict(variant), "selected_development_loss": best_dev,
        "selected_loss_profile": selected_profile, "profile_trials": [{"development_loss": trial[0], "profile": trial[1]["profile"], "history": trial[1]["history"]} for trial in trials],
        "gold_finetune": {"performed": bool(gold), "rows": len(gold), "history": fine_tune_history},
    }, temps


def _paraphrase(question: str) -> str:
    return question.replace("是否应成为该发行人近期基本面分析的重点？", "在近期基本面研究中是否值得优先核验？").replace("截至", "以")


def _intervention_report(model: DualHeadMicroconceptEncoder, dataset: EncodedRows, batch_size: int, device: torch.device) -> dict[str, Any]:
    """Run preregistered evidence removal, sign flip, and q/g paraphrase checks."""
    model.eval()
    removed_deltas: list[float] = []; flipped_deltas: list[float] = []; evidence_invariance: list[float] = []; relevance_paraphrase: list[float] = []
    with torch.no_grad():
        for raw in _loader(dataset, batch_size):
            batch = {key: value.to(device) for key, value in raw.items() if isinstance(value, torch.Tensor)}
            original = select_node_logits(model(batch), batch["concept"])
            original_status = torch.softmax(original["status"], dim=-1)
            observed = batch["status"]
            known = observed != IGNORE
            # Evidence deletion: mask all numeric values and all structured claims.
            removed = {key: value.clone() for key, value in batch.items()}
            half = removed["financial"].shape[1] // 2
            removed["financial"][:, :half] = 0.0; removed["financial"][:, half:] = 1.0
            for key in ("claim_id", "polarity", "horizon", "strength", "confidence"):
                removed[key].zero_()
            removed_output = select_node_logits(model(removed), batch["concept"])
            removed_status = torch.softmax(removed_output["status"], dim=-1)
            if bool(known.any()):
                idx = torch.arange(len(observed), device=device)[known]
                removed_deltas.extend((original_status[idx, observed[idx]] - removed_status[idx, observed[idx]]).detach().cpu().tolist())
            # Counterfactual sign reversal: flip normalised numeric values and
            # claim polarity positive<->negative.  Expected registered direction:
            # confidence in the original observed status decreases.
            flipped = {key: value.clone() for key, value in batch.items()}
            flipped["financial"][:, :half] *= -1
            p = flipped["polarity"]; flipped["polarity"] = torch.where(p == 1, torch.tensor(2, device=device), torch.where(p == 2, torch.tensor(1, device=device), p))
            flipped_status = torch.softmax(select_node_logits(model(flipped), batch["concept"])["status"], dim=-1)
            if bool(known.any()):
                idx = torch.arange(len(observed), device=device)[known]
                flipped_deltas.extend((original_status[idx, observed[idx]] - flipped_status[idx, observed[idx]]).detach().cpu().tolist())
            # Paraphrase q only. Evidence logits of a proper dual-head model are
            # architecturally invariant; relevance should move little.
            paraphrase = {key: value.clone() for key, value in batch.items()}
            paraphrase["text"] = torch.from_numpy(np.stack([__import__("research.phase1.microconcept_encoder_v1.data", fromlist=["frozen_hash_text"]).frozen_hash_text(_paraphrase(ex.question), ex.goal) for ex in raw["examples"]])).to(device)
            paraphrased = select_node_logits(model(paraphrase), batch["concept"])
            evidence_invariance.extend(torch.abs(torch.softmax(original["status"], -1) - torch.softmax(paraphrased["status"], -1)).amax(dim=-1).detach().cpu().tolist())
            r0 = (torch.softmax(original["relevance"], -1) * torch.arange(4, device=device)).sum(dim=-1)
            r1 = (torch.softmax(paraphrased["relevance"], -1) * torch.arange(4, device=device)).sum(dim=-1)
            relevance_paraphrase.extend(torch.abs(r0-r1).detach().cpu().tolist())
    def summarize(values: list[float]) -> dict[str, Any]:
        return {"n": len(values), "mean_delta": float(np.mean(values)) if values else None, "expected_direction_rate": float(np.mean(np.asarray(values) > 0)) if values else None}
    return {
        "preregistration": {
            "evidence_deletion": "For observed supported/contradicted labels, deleting all F and C should lower probability of that observed evidence-status class.",
            "counterfactual_flip": "Flipping numeric signs and positive/negative structured-claim polarity should lower probability of the originally observed evidence-status class.",
            "question_paraphrase": "Evidence-state probabilities must remain invariant in a separated dual-head model; relevance expected-score change should be small.",
            "goal_change": "Focused tasks should have larger predicted relevance than paired unrelated tasks; measured separately from labels.",
        },
        "evidence_deletion": summarize(removed_deltas), "counterfactual_evidence_flip": summarize(flipped_deltas),
        "question_paraphrase": {"n": len(evidence_invariance), "evidence_max_probability_delta_mean": float(np.mean(evidence_invariance)) if evidence_invariance else None, "relevance_expected_score_abs_delta_mean": float(np.mean(relevance_paraphrase)) if relevance_paraphrase else None},
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def _evidence_only_metrics(metrics: dict[str, Any], variant: ModelVariant) -> dict[str, Any]:
    if not variant.evidence_only:
        return metrics
    redacted = deepcopy(metrics)
    redacted["applicability"] = {"not_trained": True, "reason": "F+C baseline is intentionally evidence-only."}
    redacted["relevance"] = {"not_trained": True, "reason": "F+C baseline is intentionally evidence-only."}
    for concept in redacted.get("per_concept", {}).values():
        concept["relevance"] = {"not_trained": True, "reason": "F+C baseline is intentionally evidence-only."}
    return redacted


def _timestamp_anomalies(paths: Iterable[Path]) -> list[dict[str, Any]]:
    # The task context supplies a calendar date (2026-07-19), not a midnight
    # timestamp.  Only a later calendar day is anomalous; files written later on
    # the same day are not falsely classified as future information.
    current_date = datetime(2026, 7, 19, tzinfo=timezone.utc).date()
    result: list[dict[str, Any]] = []
    for path in paths:
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified.date() > current_date:
            result.append({"path": str(path), "mtime_utc": modified.isoformat(), "relative_to_task_current_date": "future_calendar_day"})
    return result


def run(args: argparse.Namespace) -> Path:
    candidates = Path(args.candidates); claims = Path(args.claims); labels_dir = Path(args.labels)
    output_root = Path(args.output_root); run_dir = output_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"run directory exists: {run_dir}; use --overwrite after preserving the prior run")
        shutil.rmtree(run_dir)
    config = _load_config(Path(args.config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    _set_seed(int(config["seed"])); torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = _device(args.device)
    data = build_encoder_data(candidates, claims, labels_dir)
    splits = make_splits(data, float(config["split"]["issuer_holdout_fraction"]), str(config["split"]["split_seed"]))
    train_rows = list(splits.train); dev_rows = list(splits.development); issuer_rows = list(splits.issuer_heldout)
    if not train_rows or not dev_rows:
        raise RuntimeError("default temporal 2017–2021 split lacks labeled train/development rows")
    normalizer = TrainOnlyNormalizer.fit(train_rows)
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "run_config.json", config)
    atomic_write_json(run_dir / "source_audit.json", data.source_audit)
    atomic_write_json(run_dir / "split_audit.json", splits.audit)
    atomic_write_jsonl(run_dir / "split_membership.jsonl", serializable_split_rows(splits))
    atomic_write_json(run_dir / "normalizer_train_only.json", normalizer.to_dict())
    checkpoints = run_dir / "checkpoints"; predictions_dir = run_dir / "predictions"; reports = run_dir / "reports"
    checkpoints.mkdir(); predictions_dir.mkdir(); reports.mkdir()
    results: dict[str, Any] = {}; intervention_results: dict[str, Any] = {}
    # The deterministic rule projection must remain visible; it will often win on
    # labels created by the same rule family, which is a limitation not a success.
    rule_dev = rule_projection_rows(dev_rows); rule_issuer = rule_projection_rows(issuer_rows)
    rule_temps = select_temperatures(rule_dev, config["calibration"]["temperature_grid"])
    results["rule_projection"] = {
        "training": {"not_learned": True, "reason": "recomputed deterministic F+C projection and q/g task template"},
        "temperatures": rule_temps.to_dict(), "development": evaluate_predictions(rule_dev, rule_temps, "development", "Weak/Silver exploratory"),
        "issuer_heldout": evaluate_predictions(rule_issuer, rule_temps, "issuer_heldout", "Weak/Silver exploratory"),
        "goal_sensitivity": task_pair_metrics(rule_dev, rule_temps),
    }
    atomic_write_jsonl(predictions_dir / "rule_projection_development.jsonl", rule_dev)
    atomic_write_jsonl(predictions_dir / "rule_projection_issuer_heldout.jsonl", rule_issuer)
    for variant_index, variant in enumerate(VARIANTS):
        model, training_report, temps = _fit_variant(variant, train_rows, dev_rows, normalizer, config, device, int(config["seed"]) + 100 * variant_index, checkpoints)
        text_dim = int(config["architecture"]["text_dim"]); batch_size = int(config["training"]["batch_size"])
        dev_set = EncodedRows(dev_rows, normalizer, text_dim, False, int(config["seed"]))
        issuer_set = EncodedRows(issuer_rows, normalizer, text_dim, False, int(config["seed"]))
        dev_predictions = prediction_rows(model, _loader(dev_set, batch_size), device, dev_rows)
        issuer_predictions = prediction_rows(model, _loader(issuer_set, batch_size), device, issuer_rows)
        results[variant.name] = {
            "training": training_report, "temperatures": temps.to_dict(),
            "development": _evidence_only_metrics(evaluate_predictions(dev_predictions, temps, "development", "Weak/Silver exploratory"), variant),
            "issuer_heldout": _evidence_only_metrics(evaluate_predictions(issuer_predictions, temps, "issuer_heldout", "Weak/Silver exploratory"), variant),
            "goal_sensitivity": {"not_trained": True, "reason": "F+C baseline is intentionally evidence-only."} if variant.evidence_only else task_pair_metrics(dev_predictions, temps),
        }
        intervention_results[variant.name] = _intervention_report(model, dev_set, batch_size, device)
        atomic_write_jsonl(predictions_dir / f"{variant.name}_development.jsonl", dev_predictions)
        atomic_write_jsonl(predictions_dir / f"{variant.name}_issuer_heldout.jsonl", issuer_predictions)
    gold_count = sum(1 for row in data.examples if row.effective_tier == "Gold")
    sealed_labeled_count = splits.audit["labeled_sealed_targets"]
    sealed_status = {
        "run_requested": bool(args.run_sealed_test), "executed": False,
        "reason": "No labeled 2022-2023 task rows in the supplied microconcept-labels-v1 pilot; frozen Episodes are intentionally unannotated." if sealed_labeled_count == 0 else "Gold-only sealed evaluation is not permitted because no genuine Gold labels exist.",
        "labeled_task_rows": sealed_labeled_count, "genuine_gold_rows": gold_count,
    }
    if args.run_sealed_test:
        raise RuntimeError(sealed_status["reason"])
    atomic_write_json(reports / "sealed_test_status.json", sealed_status)
    benchmark = {
        "report_scope": "exploratory_only" if gold_count == 0 else "contains_gold_but_not_sealed_gold_claim",
        "gold_policy": "No automatic label is treated as Gold. A final effectiveness claim requires an independent human/adjudicated Gold 2022–2023 test.",
        "variants": results,
        "comparison_note": "Rule projection is generated from the same deterministic label family; parity or superiority is expected and prevents any claim that the learned model has surpassed it.",
    }
    atomic_write_json(reports / "benchmark_report.json", benchmark)
    atomic_write_json(reports / "intervention_report.json", intervention_results)
    limitations = {
        "status": "exploratory_only" if gold_count == 0 else "not_final_without_independent_sealed_gold",
        "genuine_gold_rows": gold_count, "sealed_labeled_rows": sealed_labeled_count,
        "limitations": [
            "The available 400-Episode pilot contains automatic Weak/Silver labels and zero qualifying human/adjudicated Gold labels.",
            "No 2022–2023 frozen Episode is labeled in the supplied v1 label set, so the sealed temporal test was not consumed or evaluated.",
            "The available PyTorch runtime is CPU-only; the architecture is small enough for the requested 8 GB GPU but this run did not use an RTX GPU.",
            "The frozen text encoder is deterministic hash n-grams, not a pretrained semantic transformer; it is deliberately lightweight and should be replaced only after Gold data exists.",
            "The deterministic rule baseline shares the automatic-label construction family, so exploratory comparison cannot establish learned generalisation.",
            "Input file modification timestamps after 2026-07-19 are recorded as anomalies and were not used to relax availability or split boundaries.",
        ],
    }
    atomic_write_json(reports / "limitations.json", limitations)
    # Validate the freshly written artifacts before sealing their hashes in the
    # manifest.  The standalone validator repeats this check after completion.
    from .validate_run import validate_run
    internal_validation = validate_run(run_dir, verify_manifest=False)
    atomic_write_json(reports / "validation_report.json", internal_validation)
    if not internal_validation["valid"]:
        raise RuntimeError(f"run artifact validation failed: {internal_validation['errors']}")
    input_paths = [candidates, claims, labels_dir / "episode_microconcept_labels.jsonl", labels_dir / "task_relevance_labels.jsonl"]
    manifest = {
        "run_version": "microconcept-encoder-v1.0.0", "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "task_current_date": "2026-07-19",
        "data_fingerprints": data.source_fingerprints, "split_audit": splits.audit, "environment": _environment(),
        "random_seed": int(config["seed"]), "device": str(device), "label_tiers": data.source_audit["source_tiers"],
        "genuine_gold_rows": gold_count, "sealed_test": sealed_status,
        "time_metadata_anomalies": _timestamp_anomalies(input_paths),
        "prohibited_model_inputs": data.source_audit["forbidden_model_inputs"],
        "outputs": {},
    }
    # Hash outputs only after all artifacts except manifest itself are complete.
    from .data import sha256_file
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(run_dir))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    atomic_write_json(run_dir / "manifest.json", manifest)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--labels", required=True, help="Directory containing the v1 JSONL labels")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Explicit exploratory override recorded in run_config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-sealed-test", action="store_true", help="Guarded; refused without eligible labeled Gold rows")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
