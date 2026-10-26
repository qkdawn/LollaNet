"""Deprecated: hash/heuristic Weak supervision is intentionally disabled.

Use ``prepare_gpt_teacher_packets_v1`` followed by ``llm_teacher_v1`` with
different external initial and audit models.  This module exists solely so old
import paths fail safely rather than silently producing synthetic labels.
"""
from __future__ import annotations
from typing import Any, Iterable
from .question_set_v1 import ValidationError

def generate_labels(episodes: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    del episodes
    raise ValidationError(
        "heuristic/hash Weak teacher is disabled; generate route-blind packets and run independent external LLM initial/audit labeling"
    )
