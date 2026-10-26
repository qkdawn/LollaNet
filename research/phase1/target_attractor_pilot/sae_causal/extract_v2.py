from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.phase1.template_attractor_pilot.pilot import render_prompt

from .contracts import assert_discovery_rows, read_jsonl
from .extract import capture_early_window
from .resources import require_commit_headroom
from .sae import QwenScopeTopKSAE, SparseCodes


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
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=12.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.trajectories)
    assert_discovery_rows(rows, expected=600)
    require_commit_headroom(args.minimum_commit_headroom_gb)
    args.output.mkdir(parents=True, exist_ok=True)
    activations_path = args.output / "layer18_early16_activations.npz"
    completed: list[str] = []
    values: list[np.ndarray] = []
    if activations_path.is_file():
        with np.load(activations_path) as old:
            completed = old["record_ids"].tolist()
            values = list(old["activations"])
    if completed != [str(row["record_id"]) for row in rows[:len(completed)]]:
        raise ValueError("activation checkpoint is not a prefix of the frozen dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.eval()
    for row in rows[len(completed):]:
        rendered = render_prompt(tokenizer, str(row["prompt"]))
        values.append(capture_early_window(
            model, tokenizer, rendered, str(row["response"]), args.layer, args.tokens,
        ))
        completed.append(str(row["record_id"]))
        if len(completed) % args.checkpoint_every == 0 or len(completed) == len(rows):
            np.savez_compressed(
                activations_path, record_ids=np.asarray(completed),
                activations=np.asarray(values, dtype=np.float32),
            )
            print(f"captured_v2 {len(completed)}/{len(rows)}", flush=True)

    sae = QwenScopeTopKSAE.load(args.sae, top_k=args.top_k)
    indices, code_values = sae.encode(np.asarray(values, dtype=np.float32), batch_size=args.sae_batch)
    SparseCodes(indices, code_values, completed, sae.width).save(
        args.output / "layer18_early16_sparse_codes.npz",
    )
    (args.output / "extract_manifest.json").write_text(json.dumps({
        "type": "sae_activation_extraction_v2",
        "status": "ready",
        "records": len(completed),
        "layer": args.layer,
        "tokens": args.tokens,
        "prompt_rendering": "system_user_chat_template_enable_thinking_false",
        "model_path": str(args.model.resolve()),
        "sae_width": sae.width,
        "top_k": sae.top_k,
        "supersedes": "../features",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
