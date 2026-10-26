from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from ..common import stable_hash


NO_STEERING = {"base", "causal_cot"}


@contextmanager
def steer_first_batch_row(
    model: Any, layer_index: int, vector: Any, alpha: float,
) -> Iterator[dict[str, Any]]:
    import torch

    from ..qualification.hooks import transformer_layers

    layers = transformer_layers(model)
    state: dict[str, Any] = {
        "calls": 0, "finite": True, "initial_activation": None, "enabled": True,
    }

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not state["enabled"]:
            return output
        if int(hidden.shape[0]) < 1:
            raise ValueError("sealed steering requires at least one batch row")
        if int(vector.numel()) != int(hidden.shape[-1]):
            raise ValueError("sealed steering vector hidden size differs")
        modified = hidden.clone()
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * torch.as_tensor(
            alpha, device=hidden.device, dtype=hidden.dtype,
        )
        modified[0, -1] = modified[0, -1] + delta
        if state["initial_activation"] is None:
            state["initial_activation"] = modified[0, -1].detach().to(
                device="cpu", dtype=torch.float32,
            ).contiguous()
        state["calls"] += 1
        state["finite"] = bool(
            state["finite"] and torch.isfinite(modified[0, -1]).all().item()
        )
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = layers[int(layer_index)].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def causal_cot_prompt(base_prompt: str, template_path: Path) -> str:
    guidance = template_path.read_text(encoding="utf-8").strip()
    if not guidance:
        raise ValueError("CausalCoT guidance is empty")
    return f"{base_prompt.rstrip()}\n\n{guidance}"


def normalized_equal_all(
    directions: dict[str, np.ndarray],
    families: list[str],
    layer: int,
) -> np.ndarray:
    values = np.stack([
        np.asarray(directions[f"{family}__layer_{layer}"], dtype=np.float64)
        for family in families
    ])
    mean = values.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(mean).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("equal-all direction is non-finite or zero")
    return (mean / norm).astype(np.float32)


def vector_reference(
    condition_id: str,
    family: str,
    layer: int,
    wrong_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> tuple[str | None, str | None]:
    del wrong_mapping, shuffled_mapping
    if condition_id in NO_STEERING:
        return None, None
    if condition_id == "general_caa":
        key = f"general__layer_{layer}"
        return key, key
    if condition_id.startswith("family_direction__"):
        direction_family = condition_id.removeprefix("family_direction__")
        key = f"{direction_family}__layer_{layer}"
        return key, key
    if condition_id == "equal_all":
        key = f"equal_all__layer_{layer}"
        return key, key
    if condition_id.startswith("random_norm_matched__draw_"):
        tensor_key = f"{family}__layer_{layer}"
        return f"{condition_id}::{tensor_key}", tensor_key
    raise ValueError(f"unsupported sealed evaluation condition: {condition_id}")


def derived_source_condition(
    condition_id: str,
    family: str,
    wrong_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> str:
    if condition_id == "oracle_correct":
        direction_family = family
    elif condition_id == "oracle_wrong":
        direction_family = wrong_mapping[family]
    elif condition_id == "shuffled_mapping":
        direction_family = shuffled_mapping[family]
    else:
        raise ValueError(f"unsupported derived sealed condition: {condition_id}")
    return f"family_direction__{direction_family}"


def derive_routed_rows(
    condition_id: str,
    expected_ids: list[str],
    controller: dict[str, dict[str, Any]],
    family_direction_rows: dict[str, list[dict[str, Any]]],
    wrong_mapping: dict[str, str],
    shuffled_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    by_condition = {
        source_condition: {str(row["record_id"]): row for row in rows}
        for source_condition, rows in family_direction_rows.items()
    }
    if any(len(rows) != len(by_condition[condition]) for condition, rows in family_direction_rows.items()):
        raise ValueError("family-direction source contains duplicate records")
    derived: list[dict[str, Any]] = []
    for record_id in expected_ids:
        family = str(controller[record_id]["model_family"])
        source_condition = derived_source_condition(
            condition_id, family, wrong_mapping, shuffled_mapping,
        )
        if record_id not in by_condition.get(source_condition, {}):
            raise ValueError(f"derived condition source coverage is incomplete: {source_condition}")
        source = by_condition[source_condition][record_id]
        source_content = dict(source)
        claimed = source_content.pop("record_content_sha256", None)
        if claimed != stable_hash(source_content) or source.get("condition_id") != source_condition:
            raise ValueError(f"derived condition source row changed: {source_condition} {record_id}")
        row = {
            key: value for key, value in source.items()
            if key not in {"condition_id", "record_content_sha256"}
        }
        row.update({
            "condition_id": condition_id,
            "source_condition_id": source_condition,
            "source_record_content_sha256": claimed,
        })
        row["record_content_sha256"] = stable_hash(row)
        derived.append(row)
    return derived


def validate_mapping(mapping: dict[str, str], families: list[str], label: str) -> None:
    if set(mapping) != set(families) or set(mapping.values()) != set(families):
        raise ValueError(f"{label} is not a permutation of the frozen families")
    if any(mapping[family] == family for family in families):
        raise ValueError(f"{label} is not a derangement")


def frozen_derangement(families: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    while True:
        shuffled = [str(value) for value in rng.permutation(families)]
        if all(source != target for source, target in zip(families, shuffled)):
            return dict(zip(families, shuffled))


def validate_condition_contract(charter: dict[str, Any]) -> dict[str, Any]:
    families = list(charter["families"])
    conditions = charter["conditions"]
    validate_mapping(conditions["wrong_mapping"], families, "wrong mapping")
    validate_mapping(conditions["shuffled_mapping"], families, "shuffled mapping")
    if conditions["wrong_mapping"] == conditions["shuffled_mapping"]:
        raise ValueError("wrong and shuffled mappings must be distinct")
    if conditions["shuffled_mapping"] != frozen_derangement(
        families, int(conditions["shuffled_mapping_seed"]),
    ):
        raise ValueError("shuffled mapping differs from its frozen seed")
    random_ids = [conditions["random_primary"], *conditions["random_sensitivity"]]
    seeds = conditions["random_direction_seeds"]
    if set(seeds) != set(random_ids) or len(set(seeds.values())) != len(random_ids):
        raise ValueError("random direction draws do not have distinct frozen seeds")
    model_runs = [*conditions["deterministic_model_runs"], *random_ids]
    derived = list(conditions["derived_from_family_direction_runs"])
    all_conditions = [*model_runs, *derived]
    if len(all_conditions) != len(set(all_conditions)):
        raise ValueError("sealed evaluation conditions are duplicated")
    family_runs = {
        condition.removeprefix("family_direction__")
        for condition in conditions["deterministic_model_runs"]
        if condition.startswith("family_direction__")
    }
    if family_runs != set(families):
        raise ValueError("sealed evaluation does not run every family direction on every record")
    return {
        "condition_ids": all_conditions,
        "model_run_condition_ids": model_runs,
        "derived_condition_ids": derived,
        "deterministic_model_run_count": len(conditions["deterministic_model_runs"]),
        "random_draw_count": len(random_ids),
    }


def validate_general_equal_all(
    directions: dict[str, Any], families: list[str], layer: int, maximum_cosine: float,
) -> dict[str, float]:
    general = np.asarray(directions[f"general__layer_{layer}"], dtype=np.float64)
    equal_all = normalized_equal_all(directions, families, layer).astype(np.float64)
    frozen_equal = np.asarray(directions[f"equal_all__layer_{layer}"], dtype=np.float64)
    if not np.allclose(frozen_equal, equal_all, rtol=1e-6, atol=1e-7):
        raise ValueError("frozen equal-all tensor differs from the normalized family-direction sum")
    denominator = float(np.linalg.norm(general) * np.linalg.norm(equal_all))
    cosine = float(np.dot(general, equal_all) / denominator) if denominator > 0 else float("nan")
    if not np.isfinite(cosine) or cosine > float(maximum_cosine):
        raise ValueError("general and equal-all directions are not sufficiently distinct")
    return {"cosine_similarity": cosine, "maximum_cosine_similarity": float(maximum_cosine)}
