from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def transformer_layers(model: Any) -> Any:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise TypeError("unsupported causal-LM layer layout")


@contextmanager
def residual_probe(model: Any, layer_index: int, delta: float = 0.0) -> Iterator[dict[str, Any]]:
    import torch
    layers = transformer_layers(model)
    state: dict[str, Any] = {"calls": 0, "finite": True, "modified": False, "hidden_size": None}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        final = hidden[0, -1]
        state["calls"] += 1
        state["finite"] = bool(state["finite"] and torch.isfinite(final).all().item())
        state["hidden_size"] = int(final.shape[-1])
        if delta == 0.0:
            return output
        modified = hidden.clone()
        modified[:, -1, 0] = modified[:, -1, 0] + torch.as_tensor(delta, device=hidden.device, dtype=hidden.dtype)
        state["modified"] = True
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = layers[int(layer_index)].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()

