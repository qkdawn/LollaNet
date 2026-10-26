"""Build LollaNet's typed model graph and compile mechanism-program Context Packs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


GRAPH_OUTPUT = Path("05-Maps/model-graph.json")
OBSIDIAN_OUTPUT = Path("05-Maps/Model Graph.md")
READINESS_OUTPUT = Path("05-Maps/Mechanism Readiness.md")
MODEL_REF_RE = re.compile(r"^(mm-[a-z0-9-]+)@([1-9][0-9]*)$")
ITEM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FORMAL_RELATIONS = {"requires", "composes", "competes", "corrects", "specializes"}
DISCOVERY_RELATIONS = {"related", "contrasts"}
OBSERVATION_STATUSES = {"fact", "inference", "unknown"}
MECHANISM_INTERFACE_SECTIONS = (
    "机制接口",
    "输入",
    "转化",
    "输出",
    "反证",
)


def read_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if not match:
        raise ValueError(f"missing YAML frontmatter: {path}")
    payload = yaml.safe_load(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"frontmatter must be an object: {path}")
    return payload


def string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a string array")
    return value


def formal_relation_list(value: object) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("model_relations must be an array")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each model relation must be an object")
        relation = item.get("type")
        target = item.get("target")
        rationale = item.get("rationale")
        if relation not in FORMAL_RELATIONS:
            raise ValueError(f"unsupported formal relation: {relation}")
        if not isinstance(target, str) or MODEL_REF_RE.fullmatch(target) is None:
            raise ValueError("formal relation target must be a versioned model reference")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("formal relation needs a non-empty rationale")
        result.append({"relation": relation, "target": target, "rationale": rationale.strip()})
    return result


def mechanism_interface_status(text: str) -> dict[str, Any]:
    matches = list(re.finditer(r"^(#{2,6})\s+(.+?)\s*$", text, re.MULTILINE))
    headings = {match.group(2).strip() for match in matches}
    missing = [section for section in MECHANISM_INTERFACE_SECTIONS if section not in headings]
    empty: list[str] = []
    leaf_sections = set(MECHANISM_INTERFACE_SECTIONS) - {"机制接口"}
    for index, match in enumerate(matches):
        section = match.group(2).strip()
        if section not in leaf_sections:
            continue
        level = len(match.group(1))
        end = len(text)
        for following in matches[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        body = re.sub(r"<!--.*?-->", "", text[match.end() : end], flags=re.DOTALL)
        if not re.sub(r"[\s\-*+:`]", "", body):
            empty.append(section)
    return {
        "mechanism_readiness": "ready" if not missing and not empty else "legacy",
        "missing_interface_sections": missing,
        "empty_interface_sections": empty,
    }


def interface_issue_summary(item: dict[str, Any]) -> str:
    issues: list[str] = []
    if item["missing_interface_sections"]:
        issues.append("missing: " + ", ".join(item["missing_interface_sections"]))
    if item["empty_interface_sections"]:
        issues.append("empty: " + ", ".join(item["empty_interface_sections"]))
    return "; ".join(issues)


def collect_models(root: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "01-Models").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        data = read_frontmatter(path)
        model_id = data.get("id")
        version = data.get("version")
        if not isinstance(model_id, str) or not model_id.startswith("mm-") or not isinstance(version, int) or version < 1:
            raise ValueError(f"model card has invalid id or version: {path}")
        if model_id in models:
            raise ValueError(f"duplicate model id: {model_id}")
        try:
            related = string_list(data.get("related"))
            conflicts = string_list(data.get("conflicts"))
            relations = formal_relation_list(data.get("model_relations"))
        except ValueError as error:
            raise ValueError(f"invalid graph links in {path}: {error}") from error
        models[model_id] = {
            "id": model_id,
            "version": version,
            "name": data.get("name") or path.stem,
            "kind": data.get("kind", ""),
            "status": data.get("status", ""),
            "path": path.relative_to(root).as_posix(),
            "related": related,
            "conflicts": conflicts,
            "model_relations": relations,
            **mechanism_interface_status(text),
        }
    return models


def model_ref(model: dict[str, Any]) -> str:
    return f"{model['id']}@{model['version']}"


def build_graph(root: Path) -> dict[str, Any]:
    models = collect_models(root)
    refs = {model_ref(model): model for model in models.values()}
    name_to_id = {model["name"]: model_id for model_id, model in models.items()}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    for model_id, model in models.items():
        source = model_ref(model)
        for relation, values in (("related", model["related"]), ("contrasts", model["conflicts"])):
            for value in values:
                target_id = value if value in models else name_to_id.get(value)
                if target_id is None or target_id == model_id:
                    continue
                left, right = sorted((source, model_ref(models[target_id])))
                edges[(left, relation, right)] = {
                    "source_model": left,
                    "relation": relation,
                    "target_model": right,
                    "formal": False,
                    "rationale": "",
                }

        for item in model["model_relations"]:
            target = item["target"]
            if target not in refs:
                raise ValueError(f"formal relation from {source} references an unknown model: {target}")
            if target == source:
                raise ValueError(f"formal relation connects a model to itself: {source}")
            edge_source, edge_target = source, target
            if item["relation"] == "competes":
                edge_source, edge_target = sorted((source, target))
            key = (edge_source, item["relation"], edge_target)
            candidate = {
                "source_model": edge_source,
                "relation": item["relation"],
                "target_model": edge_target,
                "formal": True,
                "rationale": item["rationale"],
            }
            existing = edges.get(key)
            if existing and existing["formal"] and existing["rationale"] != candidate["rationale"]:
                raise ValueError(f"conflicting duplicate formal relation: {key}")
            edges[key] = candidate

    nodes = [
        {
            key: models[model_id][key]
            for key in (
                "id",
                "version",
                "name",
                "kind",
                "status",
                "path",
                "mechanism_readiness",
                "missing_interface_sections",
                "empty_interface_sections",
            )
        }
        for model_id in sorted(models)
    ]
    return {"schema_version": 2, "nodes": nodes, "edges": [edges[key] for key in sorted(edges)]}


def validate_graph(root: Path) -> list[str]:
    try:
        graph = build_graph(root)
    except ValueError as error:
        return [str(error)]
    known = {model_ref(node) for node in graph["nodes"]}
    errors: list[str] = []
    for index, edge in enumerate(graph["edges"]):
        if edge["source_model"] not in known or edge["target_model"] not in known:
            errors.append(f"edge[{index}] references an unknown model")
        if edge["source_model"] == edge["target_model"]:
            errors.append(f"edge[{index}] connects a model to itself")
        if edge["formal"] and edge["relation"] not in FORMAL_RELATIONS:
            errors.append(f"edge[{index}] has an invalid formal relation")
    return errors


def parse_model_refs(values: list[str], graph: dict[str, Any]) -> tuple[list[str], list[str]]:
    nodes = {model_ref(node): node for node in graph["nodes"]}
    normalized: list[str] = []
    errors: list[str] = []
    for value in values:
        if MODEL_REF_RE.fullmatch(value) is None or value not in nodes:
            errors.append(f"unknown model reference: {value}")
        elif nodes[value]["status"] == "deprecated":
            errors.append(f"deprecated model reference: {value}")
        elif value not in normalized:
            normalized.append(value)
    return normalized, errors


def graph_edges_for(graph: dict[str, Any], refs: set[str], *, formal_only: bool = False) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph["edges"]
        if edge["source_model"] in refs
        and edge["target_model"] in refs
        and (not formal_only or edge["formal"])
    ]


def neighbors(graph: dict[str, Any], model_refs: list[str]) -> dict[str, list[dict[str, Any]]]:
    selected, errors = parse_model_refs(model_refs, graph)
    if errors:
        raise ValueError("; ".join(errors))
    nodes = {model_ref(node): node for node in graph["nodes"]}
    result: dict[str, list[dict[str, Any]]] = {ref: [] for ref in selected}
    for edge in graph["edges"]:
        pairs = [(edge["source_model"], edge["target_model"], "out")]
        if edge["relation"] in DISCOVERY_RELATIONS or edge["relation"] == "competes":
            pairs.append((edge["target_model"], edge["source_model"], "peer"))
        else:
            pairs.append((edge["target_model"], edge["source_model"], "in"))
        for ref, other, direction in pairs:
            if ref in result:
                result[ref].append(
                    {
                        "ref": other,
                        "name": nodes[other]["name"],
                        "relation": edge["relation"],
                        "formal": edge["formal"],
                        "direction": direction,
                    }
                )
    return result


def render_trace(graph: dict[str, Any], model_refs: list[str]) -> str:
    selected, errors = parse_model_refs(model_refs, graph)
    if errors:
        raise ValueError("; ".join(errors))
    nodes = {model_ref(node): node for node in graph["nodes"]}
    edges = graph_edges_for(graph, set(selected))
    lines = ["使用的 LollaNet 模型", ""]
    lines.extend(f"- {nodes[ref]['name']} (`{ref}`)" for ref in selected)
    if edges:
        lines.extend(["", "模型图中的关系"])
        lines.extend(
            f"- {nodes[edge['source_model']]['name']} --`{edge['relation']}`--> "
            f"{nodes[edge['target_model']]['name']}"
            f"{' [formal]' if edge['formal'] else ' [discovery]'}"
            for edge in edges
        )
    return "\n".join(lines)


def _require_string_array(payload: dict[str, Any], field: str, *, nonempty: bool = False) -> list[str]:
    values = payload.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field} must be a string array")
    if nonempty and not values:
        raise ValueError(f"{field} must not be empty")
    return [value.strip() for value in values]


def _registered_edge(graph: dict[str, Any], source: str, relation: str, target: str) -> dict[str, Any] | None:
    for edge in graph["edges"]:
        if not edge["formal"] or edge["relation"] != relation:
            continue
        if edge["source_model"] == source and edge["target_model"] == target:
            return edge
        if relation == "competes" and edge["source_model"] == target and edge["target_model"] == source:
            return edge
    return None


def markdown_section(text: str, title: str) -> str | None:
    headings = list(re.finditer(r"^(#{2,6})\s+(.+?)\s*$", text, re.MULTILINE))
    for index, match in enumerate(headings):
        if match.group(2).strip() != title:
            continue
        level = len(match.group(1))
        end = len(text)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        return text[match.end() : end]
    return None


def model_excerpt(root: Path, path: str) -> str:
    section = markdown_section((root / path).read_text(encoding="utf-8"), "核心机制")
    if section is None:
        return ""
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", section)]
    return next((part for part in paragraphs if part), "")


def compile_context(graph: dict[str, Any], payload: dict[str, Any], root: Path) -> dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    as_of = payload.get("as_of", "")
    if not isinstance(as_of, str):
        raise ValueError("as_of must be a string")

    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty array")
    observation_ids: set[str] = set()
    normalized_observations: list[dict[str, str]] = []
    for item in observations:
        if not isinstance(item, dict):
            raise ValueError("each observation must be an object")
        item_id = item.get("id")
        statement = item.get("statement")
        source = item.get("source")
        status = item.get("status")
        if not isinstance(item_id, str) or ITEM_ID_RE.fullmatch(item_id) is None or item_id in observation_ids:
            raise ValueError(f"invalid or duplicate observation id: {item_id}")
        if not isinstance(statement, str) or not statement.strip() or not isinstance(source, str) or not source.strip():
            raise ValueError(f"observation {item_id} needs statement and source")
        if status not in OBSERVATION_STATUSES:
            raise ValueError(f"observation {item_id} has invalid status: {status}")
        observation_ids.add(item_id)
        normalized_observations.append({"id": item_id, "statement": statement.strip(), "source": source.strip(), "status": status})

    programs = payload.get("programs")
    if not isinstance(programs, list) or len(programs) < 2:
        raise ValueError("mechanism analysis needs at least two competing programs")
    program_ids = [item.get("id") for item in programs if isinstance(item, dict)]
    if len(program_ids) != len(programs) or any(
        not isinstance(value, str) or ITEM_ID_RE.fullmatch(value) is None for value in program_ids
    ):
        raise ValueError("each program needs a lowercase id")
    if len(set(program_ids)) != len(program_ids):
        raise ValueError("program ids must be unique")
    known_program_ids = set(program_ids)

    nodes = {model_ref(node): node for node in graph["nodes"]}
    normalized_programs: list[dict[str, Any]] = []
    used_registered_edges: list[dict[str, str]] = []
    unresolved_edges: list[dict[str, str]] = []
    call_roles: dict[str, list[dict[str, str]]] = {}

    for item in programs:
        program_id = item["id"]
        claim = item.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"program {program_id} needs a falsifiable claim")
        model_items = item.get("models")
        if not isinstance(model_items, list) or not 2 <= len(model_items) <= 5:
            raise ValueError(f"program {program_id} needs 2 to 5 models")
        refs = [model.get("ref") for model in model_items if isinstance(model, dict)]
        if len(refs) != len(model_items) or not all(isinstance(ref, str) for ref in refs):
            raise ValueError(f"program {program_id} has a model without ref")
        selected, errors = parse_model_refs(refs, graph)
        if errors:
            raise ValueError("; ".join(errors))
        if len(selected) != len(model_items):
            raise ValueError(f"program {program_id} contains duplicate model references")

        normalized_models: list[dict[str, Any]] = []
        for model_item, ref in zip(model_items, selected, strict=True):
            role = model_item.get("role")
            evidence = model_item.get("evidence")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"model {ref} in {program_id} needs a role")
            if not isinstance(evidence, list) or not evidence or not all(
                isinstance(item_id, str) and item_id in observation_ids for item_id in evidence
            ):
                raise ValueError(f"model {ref} in {program_id} needs known evidence ids")
            normalized_models.append(
                {
                    "ref": ref,
                    "name": nodes[ref]["name"],
                    "role": role.strip(),
                    "evidence": list(dict.fromkeys(evidence)),
                    "excerpt": model_excerpt(root, nodes[ref]["path"]),
                    "source": nodes[ref]["path"],
                }
            )
            call_roles.setdefault(ref, []).append({"program": program_id, "role": role.strip()})

        program_edges = item.get("edges", [])
        if not isinstance(program_edges, list):
            raise ValueError(f"program {program_id} edges must be an array")
        normalized_edges: list[dict[str, Any]] = []
        for edge in program_edges:
            if not isinstance(edge, dict):
                raise ValueError(f"program {program_id} has a malformed edge")
            source = edge.get("source")
            relation = edge.get("relation")
            target = edge.get("target")
            why = edge.get("why", "")
            if source not in selected or target not in selected or source == target:
                raise ValueError(f"program {program_id} edge must connect two selected models")
            if relation not in FORMAL_RELATIONS:
                raise ValueError(f"program {program_id} uses unsupported relation: {relation}")
            registered = _registered_edge(graph, source, relation, target)
            if registered is None and (not isinstance(why, str) or not why.strip()):
                raise ValueError(f"program {program_id} unresolved edge needs why")
            edge_result = {
                "program": program_id,
                "source": source,
                "relation": relation,
                "target": target,
                "why": registered["rationale"] if registered else why.strip(),
                "registered": registered is not None,
            }
            if registered:
                used_registered_edges.append(edge_result)
            else:
                unresolved_edges.append(edge_result)
            normalized_edges.append(edge_result)

        connected = {selected[0]}
        while True:
            expanded = connected | {
                endpoint
                for edge in normalized_edges
                if edge["source"] in connected or edge["target"] in connected
                for endpoint in (edge["source"], edge["target"])
            }
            if expanded == connected:
                break
            connected = expanded
        if connected != set(selected):
            raise ValueError(f"program {program_id} model relation graph must be connected")

        prediction = item.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"program {program_id} needs one prediction")
        statement = prediction.get("statement")
        horizon = prediction.get("horizon")
        distinguishes = prediction.get("distinguishes_from")
        if not isinstance(statement, str) or not statement.strip() or not isinstance(horizon, str) or not horizon.strip():
            raise ValueError(f"program {program_id} prediction needs statement and horizon")
        if not isinstance(distinguishes, str) or distinguishes not in known_program_ids or distinguishes == program_id:
            raise ValueError(f"program {program_id} prediction must distinguish another program")
        falsifier = item.get("falsifier")
        if not isinstance(falsifier, str) or not falsifier.strip():
            raise ValueError(f"program {program_id} needs a falsifier")

        normalized_programs.append(
            {
                "id": program_id,
                "claim": claim.strip(),
                "models": normalized_models,
                "edges": normalized_edges,
                "prediction": {
                    "statement": statement.strip(),
                    "horizon": horizon.strip(),
                    "distinguishes_from": distinguishes,
                },
                "falsifier": falsifier.strip(),
            }
        )

    call_nodes = [
        {
            "ref": ref,
            "name": nodes[ref]["name"],
            "source": nodes[ref]["path"],
            "roles": roles,
            "mechanism_readiness": nodes[ref]["mechanism_readiness"],
            "missing_interface_sections": nodes[ref]["missing_interface_sections"],
            "empty_interface_sections": nodes[ref]["empty_interface_sections"],
        }
        for ref, roles in call_roles.items()
    ]
    model_interface_warnings = [
        {
            "ref": node["ref"],
            "name": node["name"],
            "missing_interface_sections": node["missing_interface_sections"],
            "empty_interface_sections": node["empty_interface_sections"],
        }
        for node in call_nodes
        if node["mechanism_readiness"] != "ready"
    ]
    context_pack = {"question": question.strip(), "as_of": as_of.strip(), "observations": normalized_observations, "programs": normalized_programs}
    call_edges = used_registered_edges + unresolved_edges
    compilation = {
        "schema_version": 3,
        "context_pack": context_pack,
        "used_registered_edges": used_registered_edges,
        "unresolved_edges": unresolved_edges,
        "model_interface_warnings": model_interface_warnings,
        "call_subgraph": {"nodes": call_nodes, "edges": call_edges},
    }
    compilation["call_trace"] = render_program_trace(compilation)
    return compilation


def render_program_trace(compilation: dict[str, Any]) -> str:
    pack = compilation["context_pack"]
    lines = ["LollaNet mechanism trace"]
    for program in pack["programs"]:
        lines.extend(["", f"- Program `{program['id']}`: {program['claim']}"])
        lines.extend(f"  - {model['name']} (`{model['ref']}`): {model['role']}" for model in program["models"])
    if compilation["used_registered_edges"]:
        lines.append("")
        lines.append("Registered relations used")
        lines.extend(
            f"- `{edge['program']}`: `{edge['source']}` --`{edge['relation']}`--> `{edge['target']}`"
            for edge in compilation["used_registered_edges"]
        )
    if compilation["unresolved_edges"]:
        lines.append("")
        lines.append("Unresolved task-local relations")
        lines.extend(
            f"- `{edge['program']}`: `{edge['source']}` --`{edge['relation']}`--> `{edge['target']}`"
            for edge in compilation["unresolved_edges"]
        )
    if compilation["model_interface_warnings"]:
        lines.append("")
        lines.append("Legacy model interfaces")
        lines.extend(
            f"- `{warning['ref']}` {interface_issue_summary(warning)}"
            for warning in compilation["model_interface_warnings"]
        )
    return "\n".join(lines)


def render_context_pack_markdown(compilation: dict[str, Any]) -> str:
    pack = compilation["context_pack"]
    lines = [
        "# LollaNet Mechanism Context Pack",
        "",
        f"Question: {pack['question']}",
        f"As of: {pack['as_of'] or 'unspecified'}",
        "",
        "## Evidence state",
    ]
    lines.extend(
        f"- `{item['id']}` [{item['status']}]: {item['statement']} ({item['source']})"
        for item in pack["observations"]
    )
    lines.extend(["", "## Competing mechanism programs"])
    for program in pack["programs"]:
        lines.extend(["", f"### `{program['id']}`", "", program["claim"]])
        for model in program["models"]:
            lines.append(f"- {model['name']} (`{model['ref']}`): {model['role']}")
        prediction = program["prediction"]
        lines.append(f"- Prediction ({prediction['horizon']}): {prediction['statement']}")
        lines.append(f"- Falsifier: {program['falsifier']}")
    if compilation["unresolved_edges"]:
        lines.extend(["", "## Unresolved graph relations"])
        lines.extend(
            f"- `{edge['source']}` --`{edge['relation']}`--> `{edge['target']}`: {edge['why']}"
            for edge in compilation["unresolved_edges"]
        )
    if compilation["model_interface_warnings"]:
        lines.extend(["", "## Model interface warnings"])
        lines.extend(
            f"- `{warning['ref']}` {interface_issue_summary(warning)}"
            for warning in compilation["model_interface_warnings"]
        )
    return "\n".join(lines) + "\n"


def render_obsidian_index(graph: dict[str, Any]) -> str:
    nodes = {model_ref(node): node for node in graph["nodes"]}
    lines = [
        "---",
        "type: index",
        "title: LollaNet Model Graph",
        "generated_from: 01-Models",
        "---",
        "",
        "# LollaNet Model Graph",
        "",
        "> Formal edges describe reusable mechanism relations. Discovery edges only suggest cards to inspect. Neither proves a mechanism is active in a task.",
        "",
    ]
    for relation in sorted(FORMAL_RELATIONS | DISCOVERY_RELATIONS):
        relation_edges = [edge for edge in graph["edges"] if edge["relation"] == relation]
        if not relation_edges:
            continue
        label = "formal" if relation in FORMAL_RELATIONS else "discovery"
        lines.extend([f"## `{relation}` ({label})", ""])
        for edge in relation_edges:
            arrow = "<->" if relation in {"related", "contrasts", "competes"} else "->"
            lines.append(
                f"- [[{nodes[edge['source_model']]['name']}]] {arrow} [[{nodes[edge['target_model']]['name']}]]"
                + (f": {edge['rationale']}" if edge["rationale"] else "")
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def mechanism_readiness_report(graph: dict[str, Any]) -> dict[str, Any]:
    ready = [node for node in graph["nodes"] if node["mechanism_readiness"] == "ready"]
    legacy = [node for node in graph["nodes"] if node["mechanism_readiness"] != "ready"]
    return {
        "schema_version": 1,
        "total_models": len(graph["nodes"]),
        "ready_models": len(ready),
        "legacy_models": len(legacy),
        "models": [
            {
                "ref": model_ref(node),
                "name": node["name"],
                "status": node["mechanism_readiness"],
                "missing_interface_sections": node["missing_interface_sections"],
                "empty_interface_sections": node["empty_interface_sections"],
                "path": node["path"],
            }
            for node in graph["nodes"]
        ],
    }


def render_mechanism_readiness_markdown(graph: dict[str, Any]) -> str:
    report = mechanism_readiness_report(graph)
    lines = [
        "---",
        "type: index",
        "title: LollaNet Mechanism Readiness",
        "generated_from: 01-Models",
        "---",
        "",
        "# LollaNet Mechanism Readiness",
        "",
        f"- Ready: {report['ready_models']}",
        f"- Legacy: {report['legacy_models']}",
        f"- Total: {report['total_models']}",
        "",
        "> Legacy means the card has missing or empty V4 mechanism-interface sections. It is not an automatic quality judgment.",
        "",
    ]
    for item in report["models"]:
        if item["status"] == "ready":
            continue
        lines.append(f"- [[{item['name']}]] (`{item['ref']}`): {interface_issue_summary(item)}")
    return "\n".join(lines).rstrip() + "\n"


def load_json_object(path: Path) -> dict[str, Any]:
    if str(path) == "-":
        source = "stdin"
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:
            raw = sys.stdin.read()
        else:
            data = stream.read()
            try:
                raw = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                raw = data.decode(sys.stdin.encoding or "utf-8")
    else:
        source = str(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise ValueError(f"missing Context Pack input: {path}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Context Pack JSON from {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Context Pack input must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LollaNet's typed model graph and mechanism Context Packs.")
    parser.add_argument("--root", default=".", help="Vault root path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate model cards and graph references")
    audit = subparsers.add_parser("audit-mechanisms", help="Report V4 mechanism-interface readiness")
    audit.add_argument("--format", choices={"json", "markdown"}, default="json", help="Output format")
    build = subparsers.add_parser("build", help="Print the generated graph")
    build.add_argument("--apply", action="store_true", help="Retired graph-file write mode; always refused")
    neighbor_parser = subparsers.add_parser("neighbors", help="List neighboring models and typed relations")
    neighbor_parser.add_argument("--models", nargs="+", required=True, help="Model references in mm-id@version form")
    trace = subparsers.add_parser("trace", help="Show selected models and relations among them")
    trace.add_argument("--models", nargs="+", required=True, help="Model references in mm-id@version form")
    compile_parser = subparsers.add_parser("compile-context", help="Validate competing mechanism programs")
    compile_parser.add_argument("--input", required=True, help="Path to Context Pack JSON, or - for stdin")
    compile_parser.add_argument("--format", choices={"json", "markdown"}, default="json", help="Output format")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    if args.command == "build" and args.apply:
        print(
            "ERROR: graph-file persistence is retired after the PostgreSQL cutover; "
            "query lollanet.model_relations instead.",
            file=sys.stderr,
        )
        return 2
    if args.command == "validate":
        errors = validate_graph(root)
        if errors:
            print("Model graph validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Model graph: valid")
        return 0

    graph = build_graph(root)
    if args.command == "audit-mechanisms":
        if args.format == "markdown":
            print(render_mechanism_readiness_markdown(graph), end="")
        else:
            print(json.dumps(mechanism_readiness_report(graph), ensure_ascii=False, indent=2))
        return 0
    if args.command == "build":
        print(json.dumps(graph, ensure_ascii=False, indent=2))
        return 0
    if args.command == "neighbors":
        print(json.dumps(neighbors(graph, args.models), ensure_ascii=False, indent=2))
        return 0
    if args.command == "trace":
        print(render_trace(graph, args.models))
        return 0
    if args.command == "compile-context":
        compilation = compile_context(graph, load_json_object(Path(args.input)), root)
        if args.format == "markdown":
            print(render_context_pack_markdown(compilation), end="")
        else:
            print(json.dumps(compilation, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
