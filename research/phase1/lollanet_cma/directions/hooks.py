from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..qualification.hooks import transformer_layers


@contextmanager
def capture_residuals(
    model: Any, layer_indices: list[int], position: int,
) -> Iterator[dict[int, Any]]:
    """Capture one residual vector per requested layer without modifying the forward pass."""
    import torch

    layers = transformer_layers(model)
    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if layer_index in captured:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if position < 0 or position >= int(hidden.shape[1]):
                raise ValueError(f"activation position is outside the sequence: {position}")
            vector = hidden[0, position].detach().to(device="cpu", dtype=torch.float32).contiguous()
            captured[layer_index] = vector

        return hook

    for layer_index in layer_indices:
        handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()

