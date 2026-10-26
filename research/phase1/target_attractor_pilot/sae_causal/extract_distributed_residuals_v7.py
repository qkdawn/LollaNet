from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import read_jsonl
from .resources import require_commit_headroom


DEFAULT_LAYERS = (12, 18, 24, 30)
TOKEN_WINDOWS = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64))


def reconstruct_train_rows(
    packet_rows: list[dict[str, Any]], mapping: dict[str, str], labels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    label_by_id = {str(row["record_id"]): row for row in labels}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in packet_rows:
        item_id = str(item["item_id"])
        if item_id not in mapping:
            raise ValueError(f"blind item missing mapping: {item_id}")
        record_id = str(mapping[item_id])
        if record_id in seen or record_id not in label_by_id:
            raise ValueError(f"duplicate or unlabeled record: {record_id}")
        seen.add(record_id)
        label = label_by_id[record_id]
        if str(label["split"]) != "train":
            continue
        rows.append({
            "record_id": record_id,
            "case_id": str(label["case_id"]),
            "split": "train",
            "prompt": str(item["prompt"]),
            "response": str(item["response"]),
        })
    expected_train = sum(str(row["split"]) == "train" for row in labels)
    if len(seen) != len(labels) or len(rows) != expected_train:
        raise ValueError("blind packet reconstruction is incomplete")
    return sorted(rows, key=lambda row: str(row["record_id"]))


def summarize_windows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(values, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("token residuals must be rank two")
    summaries = np.zeros((len(TOKEN_WINDOWS), hidden.shape[1]), dtype=np.float32)
    mask = np.zeros(len(TOKEN_WINDOWS), dtype=bool)
    for index, (start, stop) in enumerate(TOKEN_WINDOWS):
        take_stop = min(stop, hidden.shape[0])
        if start < take_stop:
            summaries[index] = hidden[start:take_stop].mean(axis=0)
            mask[index] = True
    return summaries, mask


def _capture(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    layer_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    take = min(TOKEN_WINDOWS[-1][1], int(response_ids.shape[1]))
    combined = torch.cat((prompt_ids, response_ids[:, :take]), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    start = int(prompt_ids.shape[1])
    captured: dict[int, np.ndarray] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = (
                hidden[0, start : start + take].detach().float().cpu().numpy()
            )
            return output
        return hook

    layers = _layers(model)
    for layer_index in layer_indices:
        handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        with torch.inference_mode():
            model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_indices):
        raise RuntimeError("not all requested residual layers were captured")
    summaries, masks = [], []
    for layer_index in layer_indices:
        summary, mask = summarize_windows(captured[layer_index])
        summaries.append(summary)
        masks.append(mask)
    return np.stack(summaries), np.stack(masks), take


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    packet_rows = read_jsonl(args.packet)
    labels = read_jsonl(args.labels)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows = reconstruct_train_rows(packet_rows, mapping, labels)
    layer_indices = tuple(args.layers)
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("layer list contains duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "train_records.json"
    records = [{key: row[key] for key in ("record_id", "case_id", "split")} for row in rows]
    if records_path.is_file():
        if json.loads(records_path.read_text(encoding="utf-8")) != records:
            raise ValueError("existing residual extraction uses a different record order")
    else:
        _atomic_json(records_path, records)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    hidden_size = int(model.config.hidden_size)
    shape = (len(rows), len(layer_indices), len(TOKEN_WINDOWS), hidden_size)
    residual_path = args.output / "window_residuals.npy"
    mask_path = args.output / "window_mask.npy"
    lengths_path = args.output / "response_token_counts.npy"
    residuals = np.lib.format.open_memmap(
        residual_path, mode="r+" if residual_path.is_file() else "w+",
        dtype=np.float16, shape=shape,
    )
    masks = np.lib.format.open_memmap(
        mask_path, mode="r+" if mask_path.is_file() else "w+",
        dtype=bool, shape=shape[:-1],
    )
    lengths = np.lib.format.open_memmap(
        lengths_path, mode="r+" if lengths_path.is_file() else "w+",
        dtype=np.int16, shape=(len(rows),),
    )
    progress_path = args.output / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {
        "completed_record_ids": []
    }
    completed = set(progress["completed_record_ids"])
    for index, row in enumerate(rows):
        record_id = str(row["record_id"])
        if record_id not in completed:
            summary, mask, token_count = _capture(
                model, tokenizer, str(row["prompt"]), str(row["response"]), layer_indices
            )
            residuals[index] = summary.astype(np.float16)
            masks[index] = mask
            lengths[index] = token_count
            residuals.flush()
            masks.flush()
            lengths.flush()
            completed.add(record_id)
            progress["completed_record_ids"] = sorted(completed)
            _atomic_json(progress_path, progress)
        print(f"distributed-residual {len(completed)}/{len(rows)}", flush=True)
    manifest = {
        "type": "distributed_residual_window_extraction_v7",
        "status": "ready" if len(completed) == len(rows) else "in_progress",
        "records": len(rows),
        "split": "train_only",
        "development_accessed": False,
        "formal_evaluation_accessed": False,
        "layers": list(layer_indices),
        "token_windows": [list(window) for window in TOKEN_WINDOWS],
        "hidden_size": hidden_size,
        "dtype": "float16",
        "evidence_boundary": "Representations only; no development or formal evaluation outcomes accessed.",
    }
    _atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
