from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .annotation import build_annotation_delivery
from .data import Case, build_cases
from .pilot import capture_layers, generated_metrics, relative_layers, render_prompt, response_positions, _layers


WINDOWS = ("early_4", "early_8", "early_16")


def load_natural_cases(casebank_path: Path | None = None) -> list[Case]:
    if casebank_path is None:
        return build_cases()
    rows = _read_jsonl(casebank_path)
    if not rows:
        raise ValueError("natural casebank is empty")
    cases = []
    for index, row in enumerate(rows):
        for field in ("case_id", "split", "domain", "prompt"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"natural casebank row {index} lacks {field}")
        if row["split"] not in {"train", "development", "evaluation"}:
            raise ValueError(f"natural casebank row {index} has an invalid split")
        cases.append(Case(
            case_id=str(row["case_id"]), split=str(row["split"]), domain=str(row["domain"]),
            prompt=str(row["prompt"]), claim="", reason="", action="",
            anchors=tuple(str(value) for value in row.get("anchors", [])),
            template_family="natural_only", natural_variant=0,
        ))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("natural casebank contains duplicate case IDs")
    if len({case.prompt for case in cases}) != len(cases):
        raise ValueError("natural casebank contains duplicate prompts")
    split_counts = {
        split: sum(case.split == split for case in cases)
        for split in ("train", "development", "evaluation")
    }
    if any(count < 4 for count in split_counts.values()):
        raise ValueError("natural casebank requires at least four cases in every split")
    return cases


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record_id(case_id: str, sample_index: int) -> str:
    return f"{case_id}__sample_{sample_index:02d}"


def _generation_seed(case_index: int, sample_index: int) -> int:
    return 2026072500 + case_index * 10 + sample_index


def _ordered_present_records(
    expected_ids: list[str], records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [records_by_id[record_id] for record_id in expected_ids if record_id in records_by_id]


def apply_existing_release_gate(output_dir: Path, maximum_truncation_rate: float = 0.10) -> dict[str, Any]:
    audit_path = output_dir / "trajectory_audit.json"
    manifest_path = output_dir / "run_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("existing trajectory audit and run manifest are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = float(audit["response_tokens"]["truncation_rate"])
    passed = observed <= maximum_truncation_rate
    manifest["status"] = (
        "awaiting_independent_blind_annotations" if passed else "blocked_excessive_truncation"
    )
    manifest["annotation_release_gate"] = {
        "maximum_truncation_rate": maximum_truncation_rate,
        "observed_truncation_rate": observed,
        "passed": passed,
    }
    if not passed:
        manifest.setdefault("artifacts", {})["annotation_manifest"] = {
            "type": "template_attractor_blind_annotation_delivery_v1",
            "status": "blocked_excessive_truncation",
            "legacy_packet_files_must_not_be_scored": True,
        }
    _atomic_json(manifest_path, manifest)
    return manifest


def prepare_prefix_artifact(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    trajectories_path = output_dir / "trajectories.jsonl"
    if not trajectories_path.is_file():
        raise FileNotFoundError("trajectory records are required before prefix freezing")
    records = _read_jsonl(trajectories_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prefix_rows = []
    for row in records:
        token_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
        prefix_rows.append({
            "record_id": row["record_id"],
            **{
                window: tokenizer.decode(token_ids[:int(window.removeprefix("early_"))], skip_special_tokens=True)
                for window in WINDOWS
            },
        })
    prefixes_path = output_dir / "early_text_prefixes.jsonl"
    _atomic_jsonl(prefixes_path, prefix_rows)
    artifact = {"path": str(prefixes_path.resolve()), "sha256": _sha256(prefixes_path)}
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("artifacts", {})["prefixes"] = artifact
        _atomic_json(manifest_path, manifest)
    return artifact


def collect_trajectories(
    checkpoint: Path,
    output_dir: Path,
    samples_per_case: int = 2,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
    maximum_truncation_rate: float = 0.10,
    load_in_4bit: bool = False,
    casebank_path: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if samples_per_case < 2:
        raise ValueError("natural trajectory pilot requires at least two samples per case")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "trajectories.jsonl"
    existing = _read_jsonl(records_path)
    existing_by_id = {str(row["record_id"]): row for row in existing}
    if len(existing_by_id) != len(existing):
        raise ValueError("trajectory file contains duplicate record IDs")

    cases = load_natural_cases(casebank_path)
    expected_ids = [
        _record_id(case.case_id, sample_index)
        for case in cases for sample_index in range(samples_per_case)
    ]
    if any(record_id not in expected_ids for record_id in existing_by_id):
        raise ValueError("trajectory file contains records outside the current contract")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    ordered_records: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for sample_index in range(samples_per_case):
            record_id = _record_id(case.case_id, sample_index)
            if record_id in existing_by_id:
                ordered_records.append(existing_by_id[record_id])
                continue
            seed = _generation_seed(case_index, sample_index)
            rendered = render_prompt(tokenizer, case.prompt)
            encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(input_device)
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_ids = output[0, input_ids.shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            if not response:
                raise RuntimeError(f"empty natural trajectory: {record_id}")
            metrics = generated_metrics(response, case.anchors)
            record = {
                "record_id": record_id,
                "case_id": case.case_id,
                "split": case.split,
                "domain": case.domain,
                "sample_index": sample_index,
                "seed": seed,
                "prompt": case.prompt,
                "response": response,
                "prompt_tokens": int(input_ids.shape[1]),
                "response_tokens": int(response_ids.shape[0]),
                "generation_finished": bool(response_ids.shape[0] < max_new_tokens),
                "diagnostic_rule_metrics": metrics,
            }
            ordered_records.append(record)
            existing_by_id[record_id] = record
            _atomic_jsonl(records_path, _ordered_present_records(expected_ids, existing_by_id))
            print(f"trajectory_generation {len(ordered_records)}/{len(expected_ids)} {record_id}", flush=True)

    if [str(row["record_id"]) for row in ordered_records] != expected_ids:
        raise ValueError("trajectory records are not in the frozen order")

    layer_indices = relative_layers(len(_layers(model)))
    activation_path = output_dir / "early_activations.npz"
    activation_index_path = output_dir / "activation_index.json"
    if not activation_path.is_file():
        arrays: dict[str, list[np.ndarray]] = {
            f"{window}__layer_{layer}": [] for window in WINDOWS for layer in layer_indices
        }
        for index, record in enumerate(ordered_records):
            rendered = render_prompt(tokenizer, str(record["prompt"]))
            prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
            response_ids = tokenizer(str(record["response"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
            combined = torch.cat([prompt_ids, response_ids], dim=1).to(input_device)
            positions = response_positions(int(prompt_ids.shape[1]), int(response_ids.shape[1]))
            selected_positions = {window: positions[window] for window in WINDOWS}
            with torch.inference_mode(), capture_layers(model, layer_indices, selected_positions) as captured:
                model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False)
            for window in WINDOWS:
                for layer in layer_indices:
                    arrays[f"{window}__layer_{layer}"].append(captured[window][layer])
            if (index + 1) % 8 == 0:
                print(f"trajectory_activation {index + 1}/{len(ordered_records)}", flush=True)
        np.savez_compressed(
            activation_path,
            **{key: np.stack(values).astype(np.float32) for key, values in arrays.items()},
        )
        _atomic_json(activation_index_path, {
            "type": "template_attractor_natural_activation_index_v1",
            "record_ids": expected_ids,
            "windows": list(WINDOWS),
            "layers": layer_indices,
            "activation_sha256": _sha256(activation_path),
        })
    else:
        index_value = json.loads(activation_index_path.read_text(encoding="utf-8"))
        if index_value.get("record_ids") != expected_ids or index_value.get("activation_sha256") != _sha256(activation_path):
            raise ValueError("existing natural activation artifact differs from the current records")

    unique_responses = len({str(row["response"]) for row in ordered_records})
    diagnostic_scores = [float(row["diagnostic_rule_metrics"]["formulaic_structure_score"]) for row in ordered_records]
    token_counts = [int(row["response_tokens"]) for row in ordered_records]
    audit = {
        "type": "template_attractor_natural_trajectory_audit_v1",
        "record_count": len(ordered_records),
        "case_count": len(cases),
        "samples_per_case": samples_per_case,
        "unique_response_ratio": unique_responses / len(ordered_records),
        "formulaic_structure_score": {
            "minimum": min(diagnostic_scores),
            "median": float(np.median(diagnostic_scores)),
            "maximum": max(diagnostic_scores),
            "nonzero_rate": float(np.mean(np.asarray(diagnostic_scores) > 0)),
        },
        "response_tokens": {
            "minimum": min(token_counts), "median": float(np.median(token_counts)), "maximum": max(token_counts),
            "truncation_rate": float(np.mean([not bool(row["generation_finished"]) for row in ordered_records])),
        },
        "split_counts": {
            split: sum(str(row["split"]) == split for row in ordered_records)
            for split in ("train", "development", "evaluation")
        },
    }
    _atomic_json(output_dir / "trajectory_audit.json", audit)
    prefix_artifact = prepare_prefix_artifact(checkpoint, output_dir)
    truncation_rate = float(audit["response_tokens"]["truncation_rate"])
    annotation_ready = truncation_rate <= maximum_truncation_rate
    annotation_manifest = (
        build_annotation_delivery(ordered_records, output_dir / "annotation")
        if annotation_ready else {
            "type": "template_attractor_blind_annotation_delivery_v1",
            "status": "blocked_excessive_truncation",
            "observed_truncation_rate": truncation_rate,
            "maximum_truncation_rate": maximum_truncation_rate,
        }
    )
    manifest = {
        "type": "template_attractor_natural_trajectory_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "awaiting_independent_blind_annotations"
            if annotation_ready else "blocked_excessive_truncation"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "casebank": (
            {"path": str(casebank_path.resolve()), "sha256": _sha256(casebank_path)}
            if casebank_path is not None else {"source": "built_in_24_case_diagnostic_bank"}
        ),
        "generation": {
            "samples_per_case": samples_per_case, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p, "do_sample": True,
            "weight_loading": "nf4" if load_in_4bit else "bfloat16",
        },
        "candidate_layers": layer_indices,
        "early_windows": list(WINDOWS),
        "artifacts": {
            "trajectories": {"path": str(records_path.resolve()), "sha256": _sha256(records_path)},
            "activations": {"path": str(activation_path.resolve()), "sha256": _sha256(activation_path)},
            "activation_index": {"path": str(activation_index_path.resolve()), "sha256": _sha256(activation_index_path)},
            "prefixes": prefix_artifact,
            "audit": {"path": str((output_dir / "trajectory_audit.json").resolve()), "sha256": _sha256(output_dir / "trajectory_audit.json")},
            "annotation_manifest": annotation_manifest,
        },
        "annotation_release_gate": {
            "maximum_truncation_rate": maximum_truncation_rate,
            "observed_truncation_rate": truncation_rate,
            "passed": annotation_ready,
        },
        "boundary": "No trajectory label is inferred from the diagnostic rule score; independent blinded ratings are required before probe fitting.",
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect neutral-prompt natural trajectories for blinded annotation.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--casebank", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(apply_existing_release_gate(args.output)["status"])
        return
    if args.prefix_only:
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --prefix-only")
        print(prepare_prefix_artifact(args.checkpoint, args.output)["path"])
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --audit-only is used")
    result = collect_trajectories(
        args.checkpoint, args.output, samples_per_case=args.samples_per_case,
        max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit,
        casebank_path=args.casebank,
    )
    print(result["status"])


if __name__ == "__main__":
    main()
