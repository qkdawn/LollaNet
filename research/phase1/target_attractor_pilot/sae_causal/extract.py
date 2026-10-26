from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import assert_discovery_rows, read_jsonl
from .sae import QwenScopeTopKSAE, SparseCodes
from .resources import require_commit_headroom


def capture_early_window(model, tokenizer, prompt: str, response: str, layer: int, tokens: int = 16) -> np.ndarray:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full = torch.cat([prompt_ids, response_ids], dim=1).to(model.get_input_embeddings().weight.device)
    start = prompt_ids.shape[1]
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[:, start:min(start + tokens, hidden.shape[1])].detach().float().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=full, use_cache=False)
    finally:
        handle.remove()
    value = captured["value"]
    if value.shape[1] == 0:
        raise ValueError("response has no token available in the capture window")
    return value.mean(dim=1).numpy()[0]


def extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(args.trajectories)
    assert_discovery_rows(rows, expected=600)
    require_commit_headroom(args.minimum_commit_headroom_gb)
    total_records = len(rows)
    if args.record_limit is not None:
        rows = rows[:args.record_limit]
    args.output.mkdir(parents=True, exist_ok=True)
    activations_path = args.output / "layer18_early16_activations.npz"
    completed, values = [], []
    if activations_path.is_file():
        with np.load(activations_path) as old:
            completed = old["record_ids"].tolist()
            values = list(old["activations"])
    if completed != [str(row["record_id"]) for row in rows[:len(completed)]]:
        raise ValueError("activation checkpoint is not a prefix of the frozen dataset")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    load_options = {}
    if "quantization_config" not in config:
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": 0}, **load_options,
    )
    model.eval()
    for index, row in enumerate(rows[len(completed):], start=len(completed)):
        values.append(capture_early_window(model, tokenizer, row["prompt"], row["response"], args.layer, args.tokens))
        completed.append(str(row["record_id"]))
        if len(completed) % args.checkpoint_every == 0 or len(completed) == len(rows):
            np.savez_compressed(activations_path, record_ids=np.asarray(completed), activations=np.asarray(values, dtype=np.float32))
            print(f"captured {len(completed)}/{len(rows)}", flush=True)
    sae = QwenScopeTopKSAE.load(args.sae, top_k=args.top_k)
    indices, code_values = sae.encode(np.asarray(values, dtype=np.float32), batch_size=args.sae_batch)
    SparseCodes(indices, code_values, completed, sae.width).save(args.output / "layer18_early16_sparse_codes.npz")
    (args.output / "extract_manifest.json").write_text(json.dumps({
        "status": "ready" if len(completed) == total_records else "partial_smoke",
        "records": len(completed), "expected_records": total_records, "layer": args.layer, "tokens": args.tokens,
        "model_path": str(args.model.resolve()),
        "sae_width": sae.width, "top_k": sae.top_k,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sae-batch", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=18.0)
    parser.add_argument("--record-limit", type=int)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
