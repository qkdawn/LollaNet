from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file, stable_hash
from .data import _normalized_category, _validate_source_row, load_source
ANNOTATOR_FIELDS = {"annotation_id", "question_id", "ranked_display_labels", "exclude", "exclusion_reason"}
FORBIDDEN_PACKET_FIELDS = {"answer", "answer_index", "gold_index", "cot_content", "src", "lure_index"}
DISPLAY_LABELS = tuple("ABCDEFGHIJ")
ANNOTATION_CONTRACT_VERSION = "lollasteer_q_annotation_contract_v1"
ANNOTATION_INSTRUCTIONS = (
    "Rank all displayed options from most to least plausible. Do not use model outputs. "
    "Mark exclude only for ambiguity, broken options, or time-sensitive/ normative content."
)


def annotation_contract(
    config: dict[str, Any], source: dict[str, Any], annotator_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    data = config["data"]
    return {
        "version": ANNOTATION_CONTRACT_VERSION,
        "seed": config["seed"],
        "source": source,
        "data": {
            "dataset": data["dataset"], "revision": data["revision"], "source_split": data["source_split"],
            "categories": [_normalized_category(value) for value in data["categories"]],
            "items_per_category": int(data["items_per_category"]),
            "annotation_candidates_per_category": int(
                data.get("annotation_candidates_per_category", int(data["items_per_category"]) + 5)
            ),
        },
        "annotators": [str(value) for value in annotator_ids],
        "candidate_selection_policy": "seeded_category_hash_order_v1",
        "display_order_policy": "seeded_per_annotator_question_hash_order_v1",
        "display_labels": list(DISPLAY_LABELS),
        "packet_instructions_sha256": stable_hash(ANNOTATION_INSTRUCTIONS),
    }


def select_annotation_candidates(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    data = config["data"]
    categories = [_normalized_category(value) for value in data["categories"]]
    final_count = int(data["items_per_category"])
    candidate_count = int(data.get("annotation_candidates_per_category", final_count + 5))
    if candidate_count < final_count:
        raise ValueError("annotation candidate count must cover final item count")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _validate_source_row(row)
        category = _normalized_category(row["category"])
        if category in categories:
            grouped[category].append(row)
    selected: list[dict[str, Any]] = []
    for category in categories:
        candidates = sorted(
            grouped[category],
            key=lambda row: stable_hash([str(config["seed"]), "annotation", category, str(row["question_id"])]),
        )
        if len(candidates) < candidate_count:
            raise ValueError(f"category {category!r} has {len(candidates)} rows; need {candidate_count}")
        selected.extend(candidates[:candidate_count])
    return selected


def _display_order(question_id: str, option_count: int, annotator_id: str, seed: Any) -> list[int]:
    return sorted(range(option_count), key=lambda index: stable_hash([str(seed), "display", annotator_id, question_id, index]))


def _packet_row(row: dict[str, Any], annotator_id: str, seed: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = str(row["question_id"])
    order = _display_order(question_id, len(row["options"]), annotator_id, seed)
    display_options = [
        {"display_label": DISPLAY_LABELS[position], "text": str(row["options"][source_index])}
        for position, source_index in enumerate(order)
    ]
    annotation_id = stable_hash(["lollasteer-q-lure", annotator_id, question_id])[:24]
    packet = {
        "annotation_id": annotation_id, "question_id": question_id,
        "question": str(row["question"]), "display_options": display_options,
        "instructions": ANNOTATION_INSTRUCTIONS,
    }
    mapping = {
        "annotation_id": annotation_id, "question_id": question_id, "annotator_id": annotator_id,
        "display_to_source_index": {DISPLAY_LABELS[position]: int(source_index) for position, source_index in enumerate(order)},
        "gold_index": int(row["answer_index"]), "category": _normalized_category(row["category"]),
    }
    if FORBIDDEN_PACKET_FIELDS & set(packet):
        raise AssertionError("evaluator field escaped annotation packet")
    return packet, mapping


def prepare_annotation_packets(
    config_path: Path,
    output_dir: Path,
    annotator_ids: tuple[str, ...] = ("annotator_a", "annotator_b", "annotator_c"),
    source_jsonl: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows, source = load_source(source_jsonl, config["data"]["dataset"], config["data"]["revision"], config["data"]["source_split"])
    candidates = select_annotation_candidates(rows, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings: list[dict[str, Any]] = []
    packet_hashes: dict[str, str] = {}
    for annotator_id in annotator_ids:
        if not annotator_id or annotator_id.strip() != annotator_id:
            raise ValueError("annotator IDs must be non-empty and whitespace-free")
        packets: list[dict[str, Any]] = []
        for row in candidates:
            packet, mapping = _packet_row(row, annotator_id, config["seed"])
            packets.append(packet)
            mappings.append(mapping)
        packet_path = output_dir / f"packet-{annotator_id}.jsonl"
        atomic_write_jsonl(packet_path, packets)
        packet_hashes[annotator_id] = sha256_file(packet_path)
    mapping_path = output_dir / "mapping-withheld.json"
    atomic_write_json(mapping_path, mappings)
    contract = annotation_contract(config, source, annotator_ids)
    manifest = {
        "type": "lollasteer_q_lure_annotation_manifest", "status": "awaiting_human_annotations",
        "config": {
            "path": str(config_path.resolve()), "sha256": sha256_file(config_path),
            "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
            "annotation_contract_sha256": stable_hash(contract),
        },
        "source": source, "seed": config["seed"], "annotators": list(annotator_ids),
        "primary_annotators": list(annotator_ids[:2]),
        "categories": [_normalized_category(value) for value in config["data"]["categories"]],
        "candidate_count": len(candidates), "candidate_count_per_category": int(config["data"].get("annotation_candidates_per_category", int(config["data"]["items_per_category"]) + 5)),
        "final_count_per_category": int(config["data"]["items_per_category"]),
        "packet_sha256": packet_hashes, "mapping_sha256": sha256_file(mapping_path),
        "target_model_outputs_used": False,
        "selector_policy": "two independent blinded rankings; disagreement requires third-annotator adjudication",
    }
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    atomic_write_json(output_dir / "annotation-manifest.json", manifest)
    (output_dir / "ANNOTATOR_INSTRUCTIONS.md").write_text(
        "# LollaSteer-Q lure annotation\n\n"
        "Rank every displayed option from most to least plausible using only the question and options. "
        "Do not inspect target-model outputs, activations, steering conditions, or prior rankings. "
        "Mark `exclude: true` only for ambiguous, broken, strongly time-sensitive, or normative items. "
        "Return one JSON object per line with `annotation_id`, `question_id`, `ranked_display_labels`, `exclude`, and `exclusion_reason`.\n",
        encoding="utf-8",
    )
    return manifest


def validate_annotation_packets(annotation_dir: Path) -> dict[str, Any]:
    import json

    manifest_path = annotation_dir / "annotation-manifest.json"
    mapping_path = annotation_dir / "mapping-withheld.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_content_sha256", None)
    if expected_manifest_hash != stable_hash(manifest):
        raise ValueError("annotation manifest content hash mismatch")
    manifest["manifest_content_sha256"] = expected_manifest_hash
    config_path = Path(str(manifest["config"]["path"]))
    config_file_matches_provenance = config_path.is_file() and sha256_file(config_path) == manifest["config"]["sha256"]
    contract_hash = manifest["config"].get("annotation_contract_sha256")
    if contract_hash is not None:
        if manifest["config"].get("annotation_contract_version") != ANNOTATION_CONTRACT_VERSION:
            raise ValueError("unsupported annotation contract version")
        if not config_path.is_file():
            raise ValueError("annotation contract config is unavailable")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        current_contract = annotation_contract(config, manifest["source"], manifest["annotators"])
        if stable_hash(current_contract) != contract_hash:
            raise ValueError("annotation-relevant config changed; regenerate packets before annotation")
    elif config_path.is_file() and not config_file_matches_provenance:
        raise ValueError("legacy annotation config hash mismatch; migrate or regenerate packets before annotation")
    if sha256_file(mapping_path) != manifest["mapping_sha256"]:
        raise ValueError("withheld mapping hash mismatch")
    mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_by_key = {(str(row["annotator_id"]), str(row["question_id"])): row for row in mappings}
    if len(mapping_by_key) != len(mappings):
        raise ValueError("duplicate annotator/question mapping")

    expected_total = int(manifest["candidate_count"])
    expected_per_category = int(manifest["candidate_count_per_category"])
    reference_ids: set[str] | None = None
    category_counts: dict[str, int] = {}
    for annotator_id in manifest["annotators"]:
        packet_path = annotation_dir / f"packet-{annotator_id}.jsonl"
        if sha256_file(packet_path) != manifest["packet_sha256"][annotator_id]:
            raise ValueError(f"packet hash mismatch for {annotator_id}")
        packets = read_jsonl(packet_path)
        if len(packets) != expected_total:
            raise ValueError(f"unexpected packet size for {annotator_id}: {len(packets)}")
        ids = [str(row["question_id"]) for row in packets]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate question_id in packet for {annotator_id}")
        if reference_ids is None:
            reference_ids = set(ids)
        elif set(ids) != reference_ids:
            raise ValueError("annotator packets contain different candidate sets")
        for packet in packets:
            if FORBIDDEN_PACKET_FIELDS & set(packet):
                raise ValueError(f"forbidden evaluator field in packet {packet['annotation_id']}")
            allowed_packet_fields = {"annotation_id", "question_id", "question", "display_options", "instructions"}
            if set(packet) != allowed_packet_fields:
                raise ValueError(f"unexpected packet fields in {packet['annotation_id']}")
            mapping = mapping_by_key.get((str(annotator_id), str(packet["question_id"])))
            if mapping is None or str(mapping["annotation_id"]) != str(packet["annotation_id"]):
                raise ValueError(f"packet/mapping identity mismatch for {packet['annotation_id']}")
            options = packet["display_options"]
            labels = [str(option["display_label"]) for option in options]
            expected_labels = list(DISPLAY_LABELS[: len(options)])
            if labels != expected_labels or set(labels) != set(mapping["display_to_source_index"]):
                raise ValueError(f"incomplete display permutation for {packet['annotation_id']}")
            source_indices = [int(mapping["display_to_source_index"][label]) for label in labels]
            if sorted(source_indices) != list(range(len(options))):
                raise ValueError(f"invalid source-index permutation for {packet['annotation_id']}")
        if annotator_id == manifest["annotators"][0]:
            category_counts = defaultdict(int)
            for question_id in ids:
                category_counts[str(mapping_by_key[(str(annotator_id), question_id)]["category"])] += 1
    expected_categories = sorted(str(value) for value in manifest["categories"])
    if sorted(category_counts) != expected_categories:
        raise ValueError(f"packet categories do not match manifest: {sorted(category_counts)}")
    if any(count != expected_per_category for count in category_counts.values()):
        raise ValueError(f"unexpected category counts: {category_counts}")
    if sum(category_counts.values()) != expected_total:
        raise ValueError("category counts do not cover the candidate set")
    return {
        "status": "valid",
        "candidate_count": expected_total,
        "category_counts": dict(sorted(category_counts.items())),
        "annotators": list(manifest["annotators"]),
        "packet_sha256": manifest["packet_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "target_model_outputs_used": bool(manifest["target_model_outputs_used"]),
        "annotation_contract_sha256": contract_hash,
        "config_file_matches_provenance": config_file_matches_provenance,
    }


def migrate_unstarted_annotation_contract(annotation_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    import json

    manifest_path = annotation_dir / "annotation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("annotation_contract_sha256"):
        return {"status": "already_current", "manifest_path": str(manifest_path.resolve())}
    validate_annotation_packets(annotation_dir)
    annotation_files = sorted(annotation_dir.glob("annotations-*.jsonl"))
    downstream_files = sorted(annotation_dir.glob("packet-*-disagreements.jsonl"))
    if annotation_files or downstream_files:
        raise ValueError("annotation contract migration is allowed only before human annotation starts")
    active_config_path = config_path or Path(str(manifest["config"]["path"]))
    if not active_config_path.is_file():
        raise ValueError("annotation config is unavailable for migration")
    config = yaml.safe_load(active_config_path.read_text(encoding="utf-8"))
    contract = annotation_contract(config, manifest["source"], manifest["annotators"])
    manifest["config"] = {
        "path": str(active_config_path.resolve()), "sha256": sha256_file(active_config_path),
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        "annotation_contract_sha256": stable_hash(contract),
    }
    manifest.pop("prompt_template_sha256", None)
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    atomic_write_json(manifest_path, manifest)
    validation = validate_annotation_packets(annotation_dir)
    return {
        "status": "migrated", "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "annotation_contract_sha256": validation["annotation_contract_sha256"],
        "packet_sha256": validation["packet_sha256"], "mapping_sha256": validation["mapping_sha256"],
    }


def prepare_primary_annotation_delivery(annotation_dir: Path, output_dir: Path) -> dict[str, Any]:
    import json

    validation = validate_annotation_packets(annotation_dir)
    manifest_path = annotation_dir / "annotation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_ids = [str(value) for value in manifest.get("primary_annotators", [])]
    if len(primary_ids) != 2:
        raise ValueError("annotation manifest must identify exactly two primary annotators")
    if output_dir.resolve() == annotation_dir.resolve() or annotation_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("delivery output must be outside the source annotation directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    deliveries: dict[str, Any] = {}
    for annotator_id in primary_ids:
        source_packet = annotation_dir / f"packet-{annotator_id}.jsonl"
        destination = output_dir / annotator_id
        destination.mkdir(parents=True, exist_ok=True)
        unexpected = [path for path in destination.iterdir() if path.name not in {"packet.jsonl", "INSTRUCTIONS.md"}]
        if unexpected:
            raise ValueError(f"delivery directory contains unexpected files for {annotator_id}: {unexpected}")
        packet_path = destination / "packet.jsonl"
        packet_path.write_bytes(source_packet.read_bytes())
        instructions_path = destination / "INSTRUCTIONS.md"
        instructions_path.write_text(
            "# Independent blinded lure ranking\n\n"
            f"Annotator ID: `{annotator_id}`\n\n"
            "Rank all options for every item from most to least plausible. Work independently. "
            "Do not inspect another annotator's packet or annotations, withheld mappings, answer labels, "
            "target-model outputs, activations, or steering results. Use `exclude <reason>` only for "
            "ambiguous, broken, strongly time-sensitive, or normative items.\n",
            encoding="utf-8",
        )
        deliveries[annotator_id] = {
            "directory": str(destination.resolve()),
            "packet": {"path": str(packet_path.resolve()), "sha256": sha256_file(packet_path)},
            "instructions": {"path": str(instructions_path.resolve()), "sha256": sha256_file(instructions_path)},
        }
    delivery_manifest = {
        "type": "lollasteer_q_primary_annotation_delivery_v1", "status": "ready_for_independent_humans",
        "annotation_manifest_sha256": sha256_file(manifest_path),
        "annotation_contract_sha256": validation["annotation_contract_sha256"],
        "primary_annotators": primary_ids, "deliveries": deliveries,
        "contains_withheld_mapping": False, "contains_evaluator_labels": False,
        "contains_target_model_outputs": False, "contains_activations": False,
    }
    delivery_manifest["manifest_content_sha256"] = stable_hash(delivery_manifest)
    delivery_manifest_path = output_dir / "delivery-manifest.json"
    atomic_write_json(delivery_manifest_path, delivery_manifest)
    delivery_manifest["manifest_path"] = str(delivery_manifest_path.resolve())
    return delivery_manifest


def validate_primary_annotation_delivery(output_dir: Path) -> dict[str, Any]:
    import json
    from .annotate_lures import validate_annotation_progress

    manifest_path = output_dir / "delivery-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("manifest_content_sha256", None)
    if expected_hash != stable_hash(manifest):
        raise ValueError("annotation delivery manifest content hash mismatch")
    forbidden_markers = ("mapping", "evaluator", "outputs", "activations", "answer")
    all_files = [path for path in output_dir.rglob("*") if path.is_file() and path != manifest_path]
    forbidden = [path for path in all_files if any(marker in path.name.lower() for marker in forbidden_markers)]
    if forbidden:
        raise ValueError(f"annotation delivery contains forbidden artifacts: {forbidden}")
    progress: dict[str, Any] = {}
    for annotator_id in manifest["primary_annotators"]:
        delivery = manifest["deliveries"][annotator_id]
        directory = output_dir / annotator_id
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        required_files = {"packet.jsonl", "INSTRUCTIONS.md"}
        if not required_files <= actual_files or not (actual_files - required_files) <= {"annotations.jsonl"}:
            raise ValueError(f"unexpected delivery files for {annotator_id}: {sorted(actual_files)}")
        if sha256_file(directory / "packet.jsonl") != delivery["packet"]["sha256"]:
            raise ValueError(f"delivery packet hash mismatch for {annotator_id}")
        if sha256_file(directory / "INSTRUCTIONS.md") != delivery["instructions"]["sha256"]:
            raise ValueError(f"delivery instructions hash mismatch for {annotator_id}")
        packets = read_jsonl(directory / "packet.jsonl")
        annotations_path = directory / "annotations.jsonl"
        annotations = read_jsonl(annotations_path) if annotations_path.is_file() else []
        completed = validate_annotation_progress(packets, annotations)
        progress[annotator_id] = {
            "completed": len(completed), "total": len(packets),
            "status": "complete" if len(completed) == len(packets) else "needs_author_input",
            "annotations_sha256": sha256_file(annotations_path) if annotations_path.is_file() else None,
        }
    return {
        "status": "valid", "primary_annotators": manifest["primary_annotators"],
        "annotation_contract_sha256": manifest["annotation_contract_sha256"],
        "contains_withheld_mapping": False, "contains_evaluator_labels": False,
        "contains_target_model_outputs": False, "contains_activations": False,
        "progress": progress,
    }


def collect_primary_annotation_delivery(delivery_dir: Path, annotation_dir: Path) -> dict[str, Any]:
    import json
    from .annotate_lures import validate_annotation_progress

    delivery_validation = validate_primary_annotation_delivery(delivery_dir)
    packet_validation = validate_annotation_packets(annotation_dir)
    if delivery_validation["annotation_contract_sha256"] != packet_validation["annotation_contract_sha256"]:
        raise ValueError("delivery and central annotation contracts differ")
    delivery_manifest_path = delivery_dir / "delivery-manifest.json"
    delivery_manifest = json.loads(delivery_manifest_path.read_text(encoding="utf-8"))
    collected: dict[str, Any] = {}
    pending: list[str] = []
    for annotator_id in delivery_manifest["primary_annotators"]:
        source_dir = delivery_dir / annotator_id
        packets = read_jsonl(source_dir / "packet.jsonl")
        source_path = source_dir / "annotations.jsonl"
        annotations = read_jsonl(source_path) if source_path.is_file() else []
        completed = validate_annotation_progress(packets, annotations)
        if len(completed) != len(packets):
            pending.append(annotator_id)
            continue
        destination = annotation_dir / f"annotations-{annotator_id}.jsonl"
        if destination.is_file() and sha256_file(destination) != sha256_file(source_path):
            raise ValueError(f"central annotation already exists with different content: {annotator_id}")
        atomic_write_jsonl(destination, annotations)
        collected[annotator_id] = {
            "source_path": str(source_path.resolve()), "source_sha256": sha256_file(source_path),
            "destination_path": str(destination.resolve()), "destination_sha256": sha256_file(destination),
            "records": len(annotations),
        }
    if pending:
        return {"status": "needs_author_input", "pending_annotators": pending, "collected": collected}
    collection = {
        "type": "lollasteer_q_primary_annotation_collection_v1", "status": "ready_for_tiebreak",
        "annotation_contract_sha256": packet_validation["annotation_contract_sha256"],
        "delivery_manifest_sha256": sha256_file(delivery_manifest_path), "collected": collected,
    }
    collection["manifest_content_sha256"] = stable_hash(collection)
    collection_path = annotation_dir / "primary-annotation-collection.json"
    atomic_write_json(collection_path, collection)
    collection["manifest_path"] = str(collection_path.resolve())
    return collection


def _load_annotation_file(path: Path, annotator_id: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if set(row) != ANNOTATOR_FIELDS:
            raise ValueError(f"{path}: fields must equal {sorted(ANNOTATOR_FIELDS)}")
        if row["annotation_id"] in result:
            raise ValueError(f"{path}: duplicate annotation_id {row['annotation_id']}")
        labels = row["ranked_display_labels"]
        if not isinstance(labels, list) or len(labels) != len(set(labels)) or not labels:
            raise ValueError(f"{path}: ranking must be a non-empty permutation")
        result[str(row["question_id"])] = {**row, "annotator_id": annotator_id}
    return result


def _source_rank(row: dict[str, Any], mapping: dict[str, Any], context: str) -> list[int]:
    display_to_source = {
        str(label): int(index) for label, index in mapping["display_to_source_index"].items()
    }
    ranking = [str(label) for label in row["ranked_display_labels"]]
    if set(ranking) != set(display_to_source):
        raise ValueError(f"ranking label set mismatch for {context}")
    return [display_to_source[label] for label in ranking]


def _top_wrong_source_index(row: dict[str, Any], mapping: dict[str, Any], context: str) -> int:
    gold_index = int(mapping["gold_index"])
    wrong_rank = [index for index in _source_rank(row, mapping, context) if index != gold_index]
    if not wrong_rank:
        raise ValueError(f"no wrong option after gold removal for {context}")
    return wrong_rank[0]


def prepare_tiebreak_packet(annotation_dir: Path, output_path: Path) -> dict[str, Any]:
    import json

    packet_validation = validate_annotation_packets(annotation_dir)
    manifest_path = annotation_dir / "annotation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_ids = [str(value) for value in manifest.get("primary_annotators", [])]
    if len(primary_ids) != 2:
        raise ValueError("annotation manifest must identify exactly two primary annotators")
    third_ids = [str(value) for value in manifest.get("annotators", []) if str(value) not in primary_ids]
    if len(third_ids) != 1:
        raise ValueError("annotation manifest must identify exactly one third annotator")
    third_id = third_ids[0]
    primary_paths = {
        annotator_id: annotation_dir / f"annotations-{annotator_id}.jsonl" for annotator_id in primary_ids
    }
    annotations = {
        annotator_id: _load_annotation_file(path, annotator_id)
        for annotator_id, path in primary_paths.items()
    }
    expected_question_ids = {
        str(row["question_id"]) for row in read_jsonl(annotation_dir / f"packet-{primary_ids[0]}.jsonl")
    }
    for annotator_id, rows in annotations.items():
        if set(rows) != expected_question_ids:
            missing = sorted(expected_question_ids - set(rows))
            extra = sorted(set(rows) - expected_question_ids)
            raise ValueError(
                f"primary annotation {annotator_id} is incomplete: missing={len(missing)}, extra={len(extra)}"
            )
    mapping_rows = json.loads((annotation_dir / "mapping-withheld.json").read_text(encoding="utf-8"))
    mapping = {(str(row["annotator_id"]), str(row["question_id"])): row for row in mapping_rows}
    third_packets = {
        str(row["question_id"]): row for row in read_jsonl(annotation_dir / f"packet-{third_id}.jsonl")
    }
    selected_ids: list[str] = []
    exclusion_disagreements: list[str] = []
    both_excluded = 0
    primary_agreements = 0
    for question_id in sorted(expected_question_ids):
        first = annotations[primary_ids[0]][question_id]
        second = annotations[primary_ids[1]][question_id]
        if bool(first["exclude"]) != bool(second["exclude"]):
            exclusion_disagreements.append(question_id)
            continue
        if bool(first["exclude"]):
            both_excluded += 1
            continue
        top_wrong_indices: list[int] = []
        for annotator_id, row in zip(primary_ids, (first, second), strict=True):
            map_row = mapping.get((annotator_id, question_id))
            if map_row is None:
                raise ValueError(f"missing withheld mapping for {annotator_id}/{question_id}")
            top_wrong_indices.append(_top_wrong_source_index(row, map_row, f"{annotator_id}/{question_id}"))
        if top_wrong_indices[0] == top_wrong_indices[1]:
            primary_agreements += 1
        else:
            selected_ids.append(question_id)
    selected_packets = [third_packets[question_id] for question_id in selected_ids]
    if output_path.resolve() == (annotation_dir / f"packet-{third_id}.jsonl").resolve():
        raise ValueError("tie-break output must not overwrite the frozen full third-annotator packet")
    atomic_write_jsonl(output_path, selected_packets)
    report = {
        "type": "lollasteer_q_tiebreak_packet_manifest",
        "status": "needs_author_input" if exclusion_disagreements else "ready",
        "annotation_manifest_sha256": sha256_file(manifest_path),
        "mapping_sha256": packet_validation["mapping_sha256"],
        "primary_annotation_sha256": {
            annotator_id: sha256_file(path) for annotator_id, path in primary_paths.items()
        },
        "source_third_packet_sha256": packet_validation["packet_sha256"][third_id],
        "target_model_outputs_used": False,
        "third_annotator": third_id,
        "counts": {
            "candidate_items": len(expected_question_ids),
            "primary_agreements": primary_agreements,
            "both_excluded": both_excluded,
            "top_wrong_choice_disagreements": len(selected_packets),
            "exclusion_disagreements": len(exclusion_disagreements),
        },
        "exclusion_disagreement_question_ids": exclusion_disagreements,
        "artifact": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
    }
    report["manifest_content_sha256"] = stable_hash(report)
    report_path = output_path.with_suffix(".manifest.json")
    atomic_write_json(report_path, report)
    report["manifest_path"] = str(report_path.resolve())
    return report


def adjudicate_lures(
    annotation_dir: Path,
    mapping_path: Path,
    output_path: Path,
    final_count_per_category: int,
    third_annotation_path: Path | None = None,
    frozen_at: str | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    import json
    packet_validation = validate_annotation_packets(annotation_dir)
    canonical_mapping_path = annotation_dir / "mapping-withheld.json"
    if mapping_path.resolve() != canonical_mapping_path.resolve():
        raise ValueError("adjudication must use the frozen mapping from annotation_dir")
    if sha256_file(mapping_path) != packet_validation["mapping_sha256"]:
        raise ValueError("adjudication mapping hash mismatch")
    manifest_path = annotation_dir / "annotation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = {(str(row["annotator_id"]), str(row["question_id"])): row for row in mapping_rows}
    annotator_ids = [str(value) for value in manifest.get("primary_annotators", [])]
    if len(annotator_ids) != 2:
        raise ValueError("exactly two primary annotators are required")
    third_ids = [str(value) for value in manifest.get("annotators", []) if str(value) not in annotator_ids]
    if len(third_ids) != 1:
        raise ValueError("exactly one third annotator is required")
    third_id = third_ids[0]
    annotations = {
        annotator_id: _load_annotation_file(annotation_dir / f"annotations-{annotator_id}.jsonl", annotator_id)
        for annotator_id in annotator_ids
    }
    third: dict[str, dict[str, Any]] = {}
    if third_annotation_path is not None:
        third = _load_annotation_file(third_annotation_path, third_id)
    question_ids = sorted(set(annotations[annotator_ids[0]]) | set(annotations[annotator_ids[1]]))
    unresolved: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for question_id in question_ids:
        rows = [annotations[annotator_id].get(question_id) for annotator_id in annotator_ids]
        if any(row is None for row in rows):
            unresolved.append({"question_id": question_id, "reason": "missing_primary_annotation"})
            continue
        first, second = rows
        if bool(first["exclude"]) != bool(second["exclude"]):
            unresolved.append({"question_id": question_id, "reason": "exclusion_disagreement"})
            continue
        if bool(first["exclude"]):
            decisions.append({"question_id": question_id, "status": "excluded", "reason": "both_annotators_excluded"})
            continue
        first_mapping = mapping.get((annotator_ids[0], question_id))
        second_mapping = mapping.get((annotator_ids[1], question_id))
        if first_mapping is None or second_mapping is None:
            raise ValueError(f"missing withheld mapping for {question_id}")
        try:
            first_wrong = _top_wrong_source_index(first, first_mapping, f"{annotator_ids[0]}/{question_id}")
            second_wrong = _top_wrong_source_index(second, second_mapping, f"{annotator_ids[1]}/{question_id}")
        except ValueError as exc:
            unresolved.append({"question_id": question_id, "reason": str(exc)})
            continue
        chosen_lure_index = first_wrong if first_wrong == second_wrong else None
        if chosen_lure_index is None:
            third_row = third.get(question_id)
            if third_row is None:
                unresolved.append({"question_id": question_id, "reason": "top_wrong_choice_disagreement_requires_third"})
                continue
            if bool(third_row["exclude"]):
                unresolved.append({"question_id": question_id, "reason": "third_annotator_excluded"})
                continue
            third_mapping = mapping.get((third_id, question_id))
            if third_mapping is None:
                unresolved.append({"question_id": question_id, "reason": "missing_third_mapping"})
                continue
            try:
                chosen_lure_index = _top_wrong_source_index(third_row, third_mapping, f"{third_id}/{question_id}")
            except ValueError as exc:
                unresolved.append({"question_id": question_id, "reason": str(exc)})
                continue
            adjudication = "third_annotator"
        else:
            adjudication = "primary_agreement"
        map_row = first_mapping
        decisions.append({"question_id": question_id, "category": map_row["category"], "status": "accepted", "lure_index": chosen_lure_index, "adjudication": adjudication})
    accepted = [row for row in decisions if row["status"] == "accepted"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_category[row["category"]].append(row)
    selected: list[dict[str, Any]] = []
    expected_categories = sorted(str(value) for value in manifest.get("categories", []))
    if not expected_categories:
        expected_categories = sorted({str(row["category"]) for row in mapping_rows if str(row["annotator_id"]) in annotator_ids})
    for category in expected_categories:
        category_rows = sorted(by_category[category], key=lambda row: stable_hash([category, row["question_id"]]))
        selected.extend(category_rows[:final_count_per_category])
    accepted_counts = {category: len(by_category.get(category, [])) for category in expected_categories}
    if unresolved or any(accepted_counts[category] < final_count_per_category for category in expected_categories):
        return {"status": "blocked", "unresolved": unresolved, "accepted_by_category": accepted_counts, "decisions": decisions}
    if frozen_at is None or not str(frozen_at).strip() or str(frozen_at) == "REPLACE_WITH_ADJUDICATION_TIMESTAMP":
        return {"status": "blocked", "reason": "frozen_at_required", "unresolved": [], "accepted_by_category": accepted_counts, "decisions": decisions}
    primary_annotation_paths = {
        annotator_id: annotation_dir / f"annotations-{annotator_id}.jsonl" for annotator_id in annotator_ids
    }
    provenance = {
        "annotation_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "mapping_sha256": sha256_file(mapping_path),
        "primary_annotation_sha256": {
            annotator_id: sha256_file(path) for annotator_id, path in primary_annotation_paths.items()
        },
        "third_annotation_sha256": sha256_file(third_annotation_path) if third_annotation_path is not None else None,
    }
    lure_payload = {
        "selector": {
            "id": "human-blinded-ranking-v1", "frozen_at": str(frozen_at),
            "method": "two_primary_annotators_third_on_disagreement",
            "uses_target_model_outputs": False, "agreement_status": "adjudicated",
            **provenance,
        },
        "items": {row["question_id"]: int(row["lure_index"]) for row in selected},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(lure_payload, sort_keys=False), encoding="utf-8")
    agreement_count = sum(1 for row in decisions if row.get("adjudication") == "primary_agreement")
    third_count = sum(1 for row in decisions if row.get("adjudication") == "third_annotator")
    report = {"status": "ready", "selected": len(selected), "unresolved": [], "accepted_by_category": accepted_counts,
              "agreement": {"primary_agreement_count": agreement_count, "third_adjudication_count": third_count},
              "provenance": provenance, "decisions": decisions, "lure_path": str(output_path)}
    resolved_report_path = report_path or output_path.with_name(f"{output_path.stem}.adjudication.json")
    atomic_write_json(resolved_report_path, report)
    report["report_path"] = str(resolved_report_path)
    return report
