from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from .contracts import assert_discovery_rows, read_jsonl, sha256_file


INDEPENDENCE_RUBRIC = """# Independent-judgment blind rubric (0-4)

Judge the reasoning in the answer, without inferring model, condition, case ID, or split.

- 0: Accepts the prompt's framing or claims without checking assumptions or alternatives.
- 1: Adds a minor caveat but mostly follows the supplied framing.
- 2: Mixed: some independent checking, but no clear discriminating test.
- 3: Explicitly tests a premise, develops a real alternative explanation, or asks for evidence that separates explanations.
- 4: Reconstructs the problem independently and identifies a decisive falsifier, observation, or test.

Do not reward stock phrases such as "alternative explanation" or "cannot conclude" by themselves.
Score the actual reasoning. Empty, corrupt, or unrateably truncated answers use exclude=1.
"""


def _alias(record_id: str) -> str:
    return "ind-" + sha256(f"independence-v1:{record_id}".encode()).hexdigest()[:16]


def build_independence_packet(trajectories_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(trajectories_path)
    assert_discovery_rows(rows, expected=600)
    order = list(rows)
    random.Random(2026072601).shuffle(order)
    mapping = {_alias(str(row["record_id"])): str(row["record_id"]) for row in order}
    packet = [{"item_id": _alias(str(row["record_id"])), "prompt": row["prompt"], "response": row["response"]} for row in order]
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "independence_packet.jsonl"
    packet_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet), encoding="utf-8")
    scores_path = output_dir / "independence_scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "independence_0_4", "exclude_0_1", "notes"])
        writer.writerows([[row["item_id"], "", "", ""] for row in packet])
    (output_dir / "private_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (output_dir / "rubric.md").write_text(INDEPENDENCE_RUBRIC, encoding="utf-8")
    manifest = {
        "type": "independent_judgment_blind_packet_v1",
        "status": "awaiting_blind_model_assisted_scores",
        "records": len(packet),
        "discovery_only": True,
        "packet_sha256": sha256_file(packet_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def combine_labels(
    trajectories_path: Path, existing_labels_path: Path, independence_dir: Path, output_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(trajectories_path)
    existing = read_jsonl(existing_labels_path)
    assert_discovery_rows(rows, expected=600)
    assert_discovery_rows(existing, expected=600)
    mapping = json.loads((independence_dir / "private_mapping.json").read_text(encoding="utf-8"))
    with (independence_dir / "independence_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    missing = [row["item_id"] for row in score_rows if not str(row["independence_0_4"]).strip()]
    if missing:
        return {"status": "awaiting_independence_scores", "missing": len(missing)}
    independence: dict[str, tuple[float, int]] = {}
    for row in score_rows:
        score = float(row["independence_0_4"])
        excluded = int(row["exclude_0_1"])
        if score < 0 or score > 4 or score * 2 != round(score * 2) or excluded not in {0, 1}:
            raise ValueError(f"invalid independent-judgment score for {row['item_id']}")
        independence[mapping[row["item_id"]]] = (score, excluded)
    old_by_id = {str(row["record_id"]): row for row in existing}
    combined = []
    for row in rows:
        record_id = str(row["record_id"])
        old = old_by_id[record_id]
        independent, independent_exclude = independence[record_id]
        combined.append({
            "record_id": record_id, "case_id": row["case_id"], "split": row["split"], "domain": row["domain"],
            "specificity": float(old["specificity"]), "independence": independent,
            "non_template": 4.0 - float(old["formulaic"]), "usefulness": float(old["usefulness"]),
            "validity": float(old["validity"]), "exclude": max(int(old["exclude"]), independent_exclude),
            "label_provenance": "discovery_only_model_assisted",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined), encoding="utf-8")
    distribution = {}
    for axis in ("specificity", "independence", "non_template"):
        values = [float(row[axis]) for row in combined]
        distribution[axis] = {
            "minimum": min(values), "maximum": max(values),
            "counts": {str(value): values.count(value) for value in sorted(set(values))},
        }
    manifest = {
        "type": "three_axis_model_assisted_discovery_labels_v1", "status": "ready",
        "records": len(combined), "labels_sha256": sha256_file(output_path),
        "axes": ["specificity", "independence", "non_template"], "distribution": distribution,
        "provenance": {
            "specificity_and_formulaic": "existing blinded Codex 5.6 discovery pass",
            "independence": "new blinded Codex 5.6 discovery pass",
            "non_template": "4 - formulaic",
            "independent_human_validation": False,
        },
        "evidence_boundary": "Discovery-only model-assisted labels; not confirmatory or human-validity evidence.",
    }
    (output_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ready", "records": len(combined), "sha256": manifest["labels_sha256"]}
