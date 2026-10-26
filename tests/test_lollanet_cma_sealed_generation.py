from __future__ import annotations

from types import SimpleNamespace

import torch

from research.phase1.lollanet_cma.sealed_eval.generation import (
    completion_mean_log_probability,
    generate_paired_steered,
    parse_option_letter,
)


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        ids = [0, 0] if text in {"prompt", "yes completion"} else [0, 1]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens):
        del skip_special_tokens
        return "Final answer: yes" if token_ids and token_ids[0] == 0 else "Final answer: no"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 2)
        torch.nn.init.zeros_(self.embedding.weight)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.config = SimpleNamespace(hidden_size=2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        del attention_mask, past_key_values, use_cache
        hidden = self.model.layers[0](self.embedding(input_ids))
        logits = torch.zeros((*hidden.shape[:2], 3), device=hidden.device)
        logits[..., 0] = hidden[..., 0]
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_paired_generation_uses_steered_row_and_reports_positive_kl(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    result = generate_paired_steered(
        _Model(), _Tokenizer(), "prompt", 10, 2, 0, torch.tensor([2.0, 0.0]), 1.0,
    )
    assert result["generated_token_ids"] == [0, 0]
    assert result["token_level_kl"] > 0
    assert result["parsed_answer"] == "yes"
    assert result["hook_calls"] == 2
    torch.testing.assert_close(torch.from_numpy(result["activation"]), torch.tensor([2.0, 0.0]))


def test_completion_score_uses_the_same_steering_rule():
    tokenizer = _Tokenizer()
    model = _Model()
    steered = completion_mean_log_probability(
        model, tokenizer, "prompt", "yes completion", 10, 0, torch.tensor([2.0, 0.0]), 1.0,
    )
    base = completion_mean_log_probability(
        model, tokenizer, "prompt", "yes completion", 10, None, None, None,
    )
    assert steered > base


def test_control_option_parser_is_frozen_and_fail_closed():
    assert parse_option_letter("reasoning\nFinal answer: C")["parsed_answer"] == "C"
    assert parse_option_letter("reasoning\nD.")["parsed_answer"] == "D"
    assert parse_option_letter("Final answer: A\nFinal answer: B")["parse_ok"] is False
    assert parse_option_letter("the answer is C")["parse_ok"] is False
