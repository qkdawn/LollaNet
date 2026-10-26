from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.phase1.template_attractor_pilot.pilot import _layers, render_prompt

from .contracts import assert_discovery_rows, read_jsonl
from .resources import require_commit_headroom
from .sae import _find_tensor


def _selected_record_ids(discovery: dict[str, Any]) -> list[str]:
    selected: set[str] = set()
    for pairs in discovery["pairs"].values():
        for pair in pairs:
            selected.add(str(pair["low_record_id"]))
            selected.add(str(pair["high_record_id"]))
    return sorted(selected)


def _capture_response_tokens(
    model: Any, tokenizer: Any, prompt: str, response: str, layer_index: int, token_limit: int
) -> np.ndarray:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    take = min(token_limit, int(response_ids.shape[1]))
    combined = torch.cat((prompt_ids, response_ids), dim=1).to(
        model.get_input_embeddings().weight.device
    )
    start = int(prompt_ids.shape[1])
    captured: dict[str, np.ndarray] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["values"] = hidden[0, start : start + take].detach().float().cpu().numpy()
        return output

    handle = _layers(model)[layer_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False)
    finally:
        handle.remove()
    if "values" not in captured or captured["values"].shape[0] != take:
        raise RuntimeError("temporal residual capture failed")
    return captured["values"].astype(np.float32)


def _gpu_topk_encode(
    residuals: np.ndarray, mask: np.ndarray, sae_path: Path, top_k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    raw = torch.load(sae_path, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    encoder = _find_tensor(state, ("w_enc", "encoder.weight"))
    bias = _find_tensor(state, ("b_enc", "encoder.bias"))
    if encoder.shape[0] == bias.numel():
        encoder = encoder.T.contiguous()
    if encoder.shape[1] != bias.numel():
        raise ValueError("unexpected SAE encoder orientation")
    width = int(bias.numel())
    del raw, state
    encoder = encoder.to(device="cuda", dtype=torch.bfloat16)
    bias = bias.to(device="cuda", dtype=torch.bfloat16)
    valid = np.flatnonzero(mask.reshape(-1))
    flat = residuals.reshape(-1, residuals.shape[-1])
    indices = np.full((flat.shape[0], top_k), -1, dtype=np.int32)
    values = np.zeros((flat.shape[0], top_k), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            positions = valid[start : start + batch_size]
            hidden = torch.as_tensor(flat[positions], device="cuda", dtype=torch.bfloat16)
            latent = hidden @ encoder + bias
            top_values, top_indices = torch.topk(latent, top_k, dim=-1)
            indices[positions] = top_indices.cpu().numpy().astype(np.int32)
            values[positions] = top_values.float().cpu().numpy().astype(np.float32)
            print(f"temporal_sae {min(start + batch_size, len(valid))}/{len(valid)}", flush=True)
    return (
        indices.reshape(*residuals.shape[:2], top_k),
        values.reshape(*residuals.shape[:2], top_k),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--token-limit", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sae-batch", type=int, default=32)
    parser.add_argument("--minimum-commit-headroom-gb", type=float, default=4.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_commit_headroom(args.minimum_commit_headroom_gb)
    trajectories = read_jsonl(args.trajectories)
    assert_discovery_rows(trajectories, expected=600)
    by_id = {str(row["record_id"]): row for row in trajectories}
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    if discovery.get("type") != "sae_paired_candidate_discovery_v3":
        raise ValueError("temporal extraction requires paired discovery v3")
    record_ids = _selected_record_ids(discovery)
    if not set(record_ids) <= set(by_id):
        raise ValueError("paired discovery references unknown records")

    args.output.mkdir(parents=True, exist_ok=True)
    shard_directory = args.output / "residual_shards"
    shard_directory.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    for index, record_id in enumerate(record_ids):
        shard = shard_directory / f"{record_id}.npy"
        if not shard.is_file():
            row = by_id[record_id]
            values = _capture_response_tokens(
                model,
                tokenizer,
                str(row["prompt"]),
                str(row["response"]),
                args.layer,
                args.token_limit,
            )
            np.save(shard, values)
        print(f"temporal_capture {index + 1}/{len(record_ids)}", flush=True)

    hidden_size = int(model.config.hidden_size)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    residuals = np.zeros((len(record_ids), args.token_limit, hidden_size), dtype=np.float32)
    mask = np.zeros((len(record_ids), args.token_limit), dtype=bool)
    for index, record_id in enumerate(record_ids):
        values = np.load(shard_directory / f"{record_id}.npy")
        residuals[index, : len(values)] = values
        mask[index, : len(values)] = True
    indices, values = _gpu_topk_encode(
        residuals, mask, args.sae, args.top_k, args.sae_batch
    )
    np.savez_compressed(
        args.output / "layer18_first16_temporal_sparse_codes.npz",
        record_ids=np.asarray(record_ids),
        indices=indices,
        values=values,
        mask=mask,
        width=np.asarray(65536),
    )
    (args.output / "manifest.json").write_text(json.dumps({
        "type": "sae_temporal_extraction_v3",
        "status": "ready",
        "records": len(record_ids),
        "layer": args.layer,
        "token_limit": args.token_limit,
        "top_k": args.top_k,
        "prompt_rendering": "system_user_chat_template_enable_thinking_false",
        "split_boundary": "train_and_development_paired_records_only",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
