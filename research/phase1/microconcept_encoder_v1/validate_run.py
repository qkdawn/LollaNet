"""Independent validator for recoverable microconcept-encoder-v1 run artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import CONCEPT_IDS, FEATURE_NAMES, sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_run(run_dir: Path, verify_manifest: bool = True) -> dict[str, Any]:
    errors: list[str] = []; warnings: list[str] = []
    required = ["run_config.json", "source_audit.json", "split_audit.json", "split_membership.jsonl", "normalizer_train_only.json", "reports/benchmark_report.json", "reports/intervention_report.json", "reports/limitations.json", "reports/sealed_test_status.json"]
    for rel in required:
        if not (run_dir / rel).is_file():
            errors.append(f"missing required artifact: {rel}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    source = json.loads((run_dir / "source_audit.json").read_text(encoding="utf-8"))
    if source.get("concept_ids") != list(CONCEPT_IDS): errors.append("fixed 11-node concept order mismatch")
    if source.get("feature_names") != list(FEATURE_NAMES): errors.append("feature contract mismatch")
    forbidden = set(source.get("forbidden_model_inputs", []))
    if not {"route_labels", "exploratory_outcome", "issuer_name", "raw_mda_text"} <= forbidden:
        errors.append("forbidden input boundary incomplete")
    normalizer = json.loads((run_dir / "normalizer_train_only.json").read_text(encoding="utf-8"))
    if normalizer.get("feature_names") != list(FEATURE_NAMES): errors.append("normalizer feature order mismatch")
    for key in ("lower", "upper", "median", "mean", "std"):
        if len(normalizer.get(key, [])) != len(FEATURE_NAMES): errors.append(f"normalizer {key} length mismatch")
    split_rows = _read_jsonl(run_dir / "split_membership.jsonl")
    allowed_splits = {"train", "development", "issuer_heldout", "sealed_temporal", "ignored"}
    if len({row.get("example_id") for row in split_rows}) != len(split_rows): errors.append("duplicate split example_id")
    if any(row.get("split") not in allowed_splits for row in split_rows): errors.append("invalid split name")
    if any(row.get("year") in {2022, 2023} and row.get("split") != "sealed_temporal" for row in split_rows): errors.append("temporal split leakage")
    train_ciks = {row.get("cik_sha256") for row in split_rows if row.get("split") == "train"}
    held_ciks = {row.get("cik_sha256") for row in split_rows if row.get("split") == "issuer_heldout"}
    if train_ciks & held_ciks: errors.append("issuer-held-out leakage")
    pred_files = sorted((run_dir / "predictions").glob("*.jsonl"))
    if len(pred_files) != 16: errors.append(f"expected 16 prediction files, got {len(pred_files)}")
    for path in pred_files:
        for row in _read_jsonl(path):
            if row.get("microconcept_id") not in CONCEPT_IDS: errors.append(f"{path.name}: unknown concept"); break
            expected = {"status": 2, "intensity": 4, "direction": 4, "applicability": 2, "relevance": 4}
            if any(len(row.get(f"logits_{key}", [])) != width for key, width in expected.items()): errors.append(f"{path.name}: logits schema mismatch"); break
            for key, maximum in (("status",1),("intensity",3),("direction",3),("applicability",1),("relevance",3)):
                value = row.get(f"target_{key}")
                if not isinstance(value, int) or not (value == -100 or 0 <= value <= maximum): errors.append(f"{path.name}: invalid masked target {key}"); break
    sealed = json.loads((run_dir / "reports" / "sealed_test_status.json").read_text(encoding="utf-8"))
    if sealed.get("executed") and (sealed.get("genuine_gold_rows", 0) <= 0): errors.append("sealed effectiveness evaluation executed without Gold")
    if not sealed.get("executed"): warnings.append("sealed temporal evaluation not executed")
    limitation = json.loads((run_dir / "reports" / "limitations.json").read_text(encoding="utf-8"))
    if limitation.get("genuine_gold_rows", 0) == 0: warnings.append("no genuine Gold rows; outputs are exploratory only")
    if verify_manifest:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file(): errors.append("missing manifest.json")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for rel, expected in manifest.get("outputs", {}).items():
                path = run_dir / rel
                if not path.is_file() or sha256_file(path) != expected.get("sha256"):
                    errors.append(f"manifest output hash mismatch: {rel}")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "checked_prediction_files": len(pred_files), "checked_split_rows": len(split_rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    report = validate_run(Path(args.run_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
