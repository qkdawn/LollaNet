from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator

import numpy as np

from .data import Case, build_cases, paired_rows


SYSTEM_PROMPT = "回答应准确、直接并针对当前问题，不要求固定格式。"
WINDOWS = ("prompt_end", "early_4", "early_8", "early_16", "full")
GENERIC_MARKERS = (
    "值得深入探讨", "多个维度", "综合考虑", "以下几个方面", "一方面", "另一方面",
    "值得注意的是", "总的来说", "当前背景下", "理论和实践意义", "从本质上看",
    "由此可见", "完整的分析框架", "通过上述分析", "不能一概而论", "辩证地看待",
    "寻求平衡", "没有简单的答案", "宏观来看", "微观来看", "全面、客观且深入",
)


def _layers(model: Any) -> Any:
    for value in (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ):
        if value is not None:
            return value
    raise TypeError("unsupported causal-LM layer layout")


def relative_layers(count: int) -> list[int]:
    return sorted({round((count - 1) * fraction) for fraction in (0.25, 0.5, 0.75)})


@contextmanager
def capture_layers(model: Any, layer_indices: list[int], positions: dict[str, int]) -> Iterator[dict[str, dict[int, np.ndarray]]]:
    captured: dict[str, dict[int, np.ndarray]] = {name: {} for name in positions}
    handles = []
    layers = _layers(model)

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            for name, position in positions.items():
                captured[name][layer_index] = hidden[0, position].detach().float().cpu().numpy()
            return output
        return hook

    try:
        for layer_index in layer_indices:
            handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def steer_layer(model: Any, layer_index: int, direction: np.ndarray, strength: float) -> Iterator[dict[str, Any]]:
    import torch
    state = {"calls": 0}
    vector = torch.as_tensor(direction)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * float(strength)
        modified = hidden.clone()
        modified[:, -1, :] = modified[:, -1, :] - delta
        state["calls"] += 1
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = _layers(model)[layer_index].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def response_positions(prompt_length: int, response_length: int) -> dict[str, int]:
    positions = {"prompt_end": prompt_length - 1, "full": prompt_length + response_length - 1}
    for count in (4, 8, 16):
        positions[f"early_{count}"] = prompt_length + min(count, response_length) - 1
    return positions


def extract_activations(model: Any, tokenizer: Any, rows: list[dict[str, object]], layer_indices: list[int]) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, list[str]]]:
    import torch
    values: dict[str, dict[int, list[np.ndarray]]] = {
        window: {layer: [] for layer in layer_indices} for window in WINDOWS
    }
    prefixes: dict[str, list[str]] = {window: [] for window in WINDOWS}
    input_device = model.get_input_embeddings().weight.device
    for index, row in enumerate(rows):
        rendered = render_prompt(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
        response_ids = tokenizer(str(row["response"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(input_device)
        positions = response_positions(int(prompt_ids.shape[1]), int(response_ids.shape[1]))
        with torch.inference_mode(), capture_layers(model, layer_indices, positions) as captured:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        for window in WINDOWS:
            for layer in layer_indices:
                values[window][layer].append(captured[window][layer])
            if window == "prompt_end":
                prefixes[window].append(str(row["prompt"]))
            elif window == "full":
                prefixes[window].append(str(row["response"]))
            else:
                count = int(window.split("_")[1])
                prefixes[window].append(tokenizer.decode(response_ids[0, :count], skip_special_tokens=True))
        if (index + 1) % 8 == 0:
            print(f"activation_capture {index + 1}/{len(rows)}", flush=True)
    stacked = {
        window: {layer: np.stack(layer_values).astype(np.float32) for layer, layer_values in layers.items()}
        for window, layers in values.items()
    }
    return stacked, prefixes


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted(set(float(value) for value in probabilities))
    return max(candidates, key=lambda value: (_metrics(y_true, probabilities, value)["balanced_accuracy"], -value))


def fit_activation_probe(train_x: np.ndarray, train_y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=20260725)).fit(train_x, train_y)


def fit_text_probe(train_text: list[str], train_y: np.ndarray) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=5000),
        LogisticRegression(C=1.0, max_iter=2000, random_state=20260725),
    ).fit(train_text, train_y)


def fit_and_select(rows: list[dict[str, object]], activations: dict[str, dict[int, np.ndarray]], prefixes: dict[str, list[str]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    split_indices = {
        split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
        for split in ("train", "development", "evaluation")
    }
    candidates = []
    fitted: dict[tuple[str, int], Any] = {}
    for window, layer_values in activations.items():
        for layer, matrix in layer_values.items():
            probe = fit_activation_probe(matrix[split_indices["train"]], labels[split_indices["train"]])
            fitted[(window, layer)] = probe
            dev_probability = probe.predict_proba(matrix[split_indices["development"]])[:, 1]
            candidates.append({"window": window, "layer": layer, **_metrics(labels[split_indices["development"]], dev_probability)})
    early_candidates = [item for item in candidates if str(item["window"]).startswith("early_")]
    selected = max(early_candidates, key=lambda item: (item["auroc"], item["auprc"], item["balanced_accuracy"], -item["layer"]))
    key = (str(selected["window"]), int(selected["layer"]))
    probe = fitted[key]
    selected_matrix = activations[key[0]][key[1]]
    dev_probability = probe.predict_proba(selected_matrix[split_indices["development"]])[:, 1]
    threshold = choose_threshold(labels[split_indices["development"]], dev_probability)
    eval_probability = probe.predict_proba(selected_matrix[split_indices["evaluation"]])[:, 1]

    text_probe = fit_text_probe(
        [prefixes[key[0]][i] for i in split_indices["train"]], labels[split_indices["train"]],
    )
    text_dev_probability = text_probe.predict_proba(
        [prefixes[key[0]][i] for i in split_indices["development"]]
    )[:, 1]
    text_threshold = choose_threshold(labels[split_indices["development"]], text_dev_probability)
    text_probability = text_probe.predict_proba([prefixes[key[0]][i] for i in split_indices["evaluation"]])[:, 1]
    lengths = np.asarray([len(prefixes[key[0]][i]) for i in range(len(rows))], dtype=np.float64).reshape(-1, 1)
    length_probe = fit_activation_probe(lengths[split_indices["train"]], labels[split_indices["train"]])
    length_dev_probability = length_probe.predict_proba(lengths[split_indices["development"]])[:, 1]
    length_threshold = choose_threshold(labels[split_indices["development"]], length_dev_probability)
    length_probability = length_probe.predict_proba(lengths[split_indices["evaluation"]])[:, 1]
    return {
        "selected": selected,
        "threshold": threshold,
        "evaluation": _metrics(labels[split_indices["evaluation"]], eval_probability, threshold),
        "text_baseline": {
            **_metrics(labels[split_indices["evaluation"]], text_probability, text_threshold),
            "threshold": text_threshold,
        },
        "length_baseline": {
            **_metrics(labels[split_indices["evaluation"]], length_probability, length_threshold),
            "threshold": length_threshold,
        },
        "candidate_development_metrics": candidates,
        "probe": probe,
        "text_probe": text_probe,
        "split_indices": split_indices,
    }


def fit_direction(rows: list[dict[str, object]], matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    by_case: dict[str, dict[int, np.ndarray]] = {}
    for index in indices:
        row = rows[int(index)]
        by_case.setdefault(str(row["case_id"]), {})[int(row["label"])] = matrix[int(index)]
    differences = [pair[1] - pair[0] for pair in by_case.values() if set(pair) == {0, 1}]
    direction = np.stack(differences).mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("template direction is degenerate")
    return (direction / norm).astype(np.float32)


def generated_metrics(text: str, anchors: tuple[str, ...]) -> dict[str, float]:
    marker_hits = sum(text.count(marker) for marker in GENERIC_MARKERS)
    numbered_sections = len(re.findall(r"(?m)^\s*\d+[.、]", text))
    bold_sections = len(re.findall(r"\*\*[^*]{1,20}\*\*", text))
    generic_scaffolds = sum(text.count(marker) for marker in ("以下是", "以下几点", "取决于具体", "综合考虑", "需要权衡"))
    return {
        "template_marker_density": marker_hits / max(len(text), 1) * 100.0,
        "formulaic_structure_score": float(marker_hits + generic_scaffolds + 0.5 * min(numbered_sections, 4) + 0.25 * min(bold_sections, 4)),
        "anchor_coverage": (
            sum(anchor.lower() in text.lower() for anchor in anchors) / len(anchors)
            if anchors else 0.0
        ),
        "characters": float(len(text)),
    }


def activation_at_prefix(model: Any, tokenizer: Any, prompt_ids: Any, prefix_ids: Any, layer: int) -> np.ndarray:
    import torch
    input_ids = torch.cat([prompt_ids, prefix_ids], dim=1)
    position = int(input_ids.shape[1]) - 1
    with torch.inference_mode(), capture_layers(model, [layer], {"gate": position}) as captured:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    return captured["gate"][layer]


def generate_condition(model: Any, tokenizer: Any, case: Case, layer: int, direction: np.ndarray, probe: Any, threshold: float, condition: str, strength: float, early_tokens: int, max_new_tokens: int) -> dict[str, Any]:
    import torch
    rendered = render_prompt(tokenizer, case.prompt)
    input_device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to(input_device)
    torch.manual_seed(20260725)
    with torch.inference_mode():
        prefix_output = model.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=early_tokens,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    prefix_ids = prefix_output[:, prompt_ids.shape[1]:]
    activation = activation_at_prefix(model, tokenizer, prompt_ids, prefix_ids, layer)
    probability = float(probe.predict_proba(activation.reshape(1, -1))[0, 1])
    gate_on = condition == "always_on" or (condition == "learned_gate" and probability >= threshold)
    if condition == "always_on":
        generation_input = prompt_ids
        remaining_tokens = max_new_tokens
    else:
        generation_input = torch.cat([prompt_ids, prefix_ids], dim=1)
        remaining_tokens = max(1, max_new_tokens - int(prefix_ids.shape[1]))
    context = steer_layer(model, layer, direction, strength) if gate_on else nullcontext({"calls": 0})
    with torch.inference_mode(), context as hook_state:
        output = model.generate(
            input_ids=generation_input, attention_mask=torch.ones_like(generation_input),
            max_new_tokens=remaining_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = output[0, prompt_ids.shape[1]:]
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return {
        "case_id": case.case_id, "condition": condition, "gate_probability": probability,
        "gate_on": gate_on, "hook_calls": int(hook_state["calls"]), "text": text,
        **generated_metrics(text, case.anchors),
    }


def run(checkpoint: Path, output: Path, smoke: bool) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(20260725)
    np.random.seed(20260725)
    cases = build_cases()
    rows = paired_rows(cases)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    layer_indices = relative_layers(len(_layers(model)))
    activations, prefixes = extract_activations(model, tokenizer, rows, layer_indices)
    probe_result = fit_and_select(rows, activations, prefixes)
    selected = probe_result["selected"]
    window = str(selected["window"])
    layer = int(selected["layer"])
    direction = fit_direction(
        rows, activations[window][layer], probe_result["split_indices"]["train"],
    )
    early_tokens = 8 if window == "prompt_end" or window == "full" else int(window.split("_")[1])
    eval_cases = [case for case in cases if case.split == "evaluation"]
    if smoke:
        eval_cases = eval_cases[:2]
    generations = []
    for case in eval_cases:
        for condition in ("baseline", "always_on", "learned_gate"):
            generations.append(generate_condition(
                model, tokenizer, case, layer, direction, probe_result["probe"],
                float(probe_result["threshold"]), condition, 3.0, early_tokens, 96,
            ))
            print(f"generation {case.case_id} {condition}", flush=True)
    summary_by_condition = {}
    for condition in ("baseline", "always_on", "learned_gate"):
        condition_rows = [row for row in generations if row["condition"] == condition]
        summary_by_condition[condition] = {
            key: float(np.mean([float(row[key]) for row in condition_rows]))
            for key in ("template_marker_density", "formulaic_structure_score", "anchor_coverage", "characters")
        }
        summary_by_condition[condition]["gate_rate"] = float(np.mean([bool(row["gate_on"]) for row in condition_rows]))
    serializable_probe = {key: value for key, value in probe_result.items() if key not in {"probe", "text_probe", "split_indices"}}
    activation_auc = float(serializable_probe["evaluation"]["auroc"])
    text_auc = float(serializable_probe["text_baseline"]["auroc"])
    baseline_formulaic = float(summary_by_condition["baseline"]["formulaic_structure_score"])
    gated_formulaic = float(summary_by_condition["learned_gate"]["formulaic_structure_score"])
    result = {
        "type": "template_attractor_pilot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_complete" if smoke else "pilot_complete",
        "checkpoint": str(checkpoint.resolve()),
        "model_layers": len(_layers(model)), "candidate_layers": layer_indices,
        "case_count": len(cases), "paired_record_count": len(rows),
        "split_case_counts": {split: sum(case.split == split for case in cases) for split in ("train", "development", "evaluation")},
        "probe": serializable_probe,
        "direction": {"layer": layer, "window": window, "l2_norm": float(np.linalg.norm(direction)), "strength": 3.0},
        "generation_summary": summary_by_condition,
        "go_no_go": {
            "early_activation_above_chance": activation_auc >= 0.70,
            "activation_beats_early_text_baseline": activation_auc > text_auc,
            "gated_caa_reduces_formulaic_score": gated_formulaic < baseline_formulaic,
            "decision": "continue_formal_pilot" if (
                activation_auc >= 0.70 and activation_auc > text_auc and gated_formulaic < baseline_formulaic
            ) else "revise_measurement_before_scaling",
        },
        "generations": generations,
        "cases": [asdict(case) for case in cases],
        "interpretation_boundary": (
            "Synthetic content-matched pilot only. Rule metrics do not replace blinded human ratings, "
            "and prompt-end activations cannot predict sampling randomness for identical prompts."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(output / "direction.npy", direction)
    print(json.dumps({"probe": serializable_probe, "generation_summary": summary_by_condition}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the template-attractor detection and steering pilot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.smoke)


if __name__ == "__main__":
    main()
