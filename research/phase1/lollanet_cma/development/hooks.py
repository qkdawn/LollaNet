from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..qualification.hooks import transformer_layers


@contextmanager
def steer_residual(model: Any, layer_index: int, vector: Any, alpha: float) -> Iterator[dict[str, Any]]:
    import torch

    layers = transformer_layers(model)
    state: dict[str, Any] = {"calls": 0, "finite": True, "hidden_size": None}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if int(vector.numel()) != int(hidden.shape[-1]):
            raise ValueError("steering vector hidden size differs from the residual stream")
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * torch.as_tensor(
            alpha, device=hidden.device, dtype=hidden.dtype,
        )
        modified = hidden.clone()
        modified[:, -1, :] = modified[:, -1, :] + delta
        state["calls"] += 1
        state["finite"] = bool(state["finite"] and torch.isfinite(modified[:, -1, :]).all().item())
        state["hidden_size"] = int(hidden.shape[-1])
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = layers[int(layer_index)].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()

