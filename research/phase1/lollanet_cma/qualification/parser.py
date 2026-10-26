from __future__ import annotations

import re
from typing import Any


FINAL_RE = re.compile(r"(?im)^\s*final\s+answer\s*:\s*(yes|no)\s*[.!]?\s*$")
TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z])(yes|no)(?![A-Za-z])")
TRAILING_RE = re.compile(r"(?i)(?<![A-Za-z])(yes|no)(?![A-Za-z])[\s.!?]*$")


def parse_yes_no(text: str) -> dict[str, Any]:
    final = [value.lower() for value in FINAL_RE.findall(text)]
    if final:
        labels = set(final)
        return {
            "parsed_answer": final[-1] if len(labels) == 1 else None,
            "parse_ok": len(labels) == 1,
            "parse_rule": "final_answer_marker" if len(labels) == 1 else "conflicting_final_markers",
        }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        trailing = [value.lower() for value in TOKEN_RE.findall(lines[-1])]
        match = TRAILING_RE.search(lines[-1])
        if match and len(set(trailing)) == 1:
            return {"parsed_answer": match.group(1).lower(), "parse_ok": True, "parse_rule": "last_nonempty_line"}
    return {
        "parsed_answer": None,
        "parse_ok": False,
        "parse_rule": "conflicting_tail_tokens" if lines and len(set(TOKEN_RE.findall(lines[-1]))) > 1 else "missing_answer",
    }
