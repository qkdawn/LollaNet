from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SparseCodes:
    indices: np.ndarray
    values: np.ndarray
    record_ids: list[str]
    width: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, indices=self.indices, values=self.values,
                            record_ids=np.asarray(self.record_ids), width=np.asarray(self.width))


def _find_tensor(state: dict[str, Any], suffixes: tuple[str, ...]) -> Any:
    for key, value in state.items():
        if any(key.lower().endswith(suffix.lower()) for suffix in suffixes):
            return value
    raise KeyError(f"SAE tensor not found for {suffixes}; keys={sorted(state)[:12]}")


class QwenScopeTopKSAE:
    def __init__(self, encoder: Any, decoder: Any, encoder_bias: Any, decoder_bias: Any, top_k: int = 50):
        self.encoder = encoder.float().cpu()
        self.decoder = decoder.float().cpu()
        self.encoder_bias = encoder_bias.float().cpu()
        self.decoder_bias = decoder_bias.float().cpu()
        self.top_k = int(top_k)
        if self.encoder.shape[0] == self.decoder_bias.numel():
            self.width = int(self.encoder.shape[1])
        else:
            self.encoder = self.encoder.T.contiguous()
            self.width = int(self.encoder.shape[1])
        if self.decoder.shape[0] != self.width:
            self.decoder = self.decoder.T.contiguous()

    @classmethod
    def load(cls, path: Path, top_k: int = 50) -> "QwenScopeTopKSAE":
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        encoder = _find_tensor(state, ("w_enc", "encoder.weight"))
        decoder = _find_tensor(state, ("w_dec", "decoder.weight"))
        encoder_bias = _find_tensor(state, ("b_enc", "encoder.bias"))
        try:
            decoder_bias = _find_tensor(state, ("b_dec", "decoder.bias"))
        except KeyError:
            decoder_bias = torch.zeros(encoder.shape[-1])
        return cls(encoder, decoder, encoder_bias, decoder_bias, top_k)

    def encode(self, activations: Any, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        import torch

        chunks_i, chunks_v = [], []
        values = torch.as_tensor(activations, dtype=torch.float32, device="cpu")
        for start in range(0, values.shape[0], batch_size):
            hidden = values[start:start + batch_size]
            latent = hidden @ self.encoder + self.encoder_bias
            top_values, top_indices = torch.topk(latent, self.top_k, dim=-1)
            chunks_i.append(top_indices.numpy().astype(np.int32))
            chunks_v.append(top_values.numpy().astype(np.float32))
        return np.concatenate(chunks_i), np.concatenate(chunks_v)

    def decoder_vector(self, feature: int) -> Any:
        return self.decoder[int(feature)].clone()
