import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_graph_module():
    path = ROOT / "06-Scripts" / "model_graph.py"
    spec = importlib.util.spec_from_file_location("model_graph", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def context_payload():
    return json.loads((ROOT / "03-Templates" / "context_pack_template.json").read_text(encoding="utf-8"))


def test_graph_contains_discovery_and_formal_mechanism_relations():
    module = load_graph_module()

    assert module.validate_graph(ROOT) == []
    graph = module.build_graph(ROOT)
    edges = {
        (edge["source_model"], edge["relation"], edge["target_model"], edge["formal"])
        for edge in graph["edges"]
    }

    assert graph["schema_version"] == 2
    assert ("mm-base-rate@3", "composes", "mm-bayesian-updating@3", True) in edges
    assert ("mm-availability-bias@3", "competes", "mm-base-rate@3", True) in edges
    assert ("mm-base-rate@3", "related", "mm-bayesian-updating@3", False) in edges
    formal = next(edge for edge in graph["edges"] if edge["relation"] == "composes")
    assert formal["rationale"]


def test_neighbors_distinguish_formal_relations_from_discovery_hints():
    module = load_graph_module()
    result = module.neighbors(module.build_graph(ROOT), ["mm-base-rate@3"])["mm-base-rate@3"]

    assert any(
        item["ref"] == "mm-bayesian-updating@3"
        and item["relation"] == "composes"
        and item["formal"] is True
        and item["direction"] == "out"
        for item in result
    )
    assert any(item["relation"] == "related" and item["formal"] is False for item in result)


def test_compiler_builds_competing_programs_and_exact_call_subgraph():
    module = load_graph_module()
    compilation = module.compile_context(module.build_graph(ROOT), context_payload(), ROOT)
    markdown = module.render_context_pack_markdown(compilation)

    assert compilation["schema_version"] == 3
    assert len(compilation["context_pack"]["programs"]) == 2
    first_model = compilation["context_pack"]["programs"][0]["models"][0]
    assert first_model["evidence"] == ["obs-1"]
    assert first_model["excerpt"]
    assert {item["relation"] for item in compilation["used_registered_edges"]} == {
        "composes",
        "competes",
    }
    assert compilation["unresolved_edges"] == []
    assert compilation["model_interface_warnings"] == []
    refs = {node["ref"] for node in compilation["call_subgraph"]["nodes"]}
    assert refs == {"mm-base-rate@3", "mm-bayesian-updating@3", "mm-availability-bias@3"}
    assert "Competing mechanism programs" in markdown
    assert "evidence-update" in compilation["call_trace"]


def test_compiler_exposes_unregistered_task_local_composition():
    module = load_graph_module()
    payload = context_payload()
    payload["programs"][0]["edges"][0]["relation"] = "requires"
    payload["programs"][0]["edges"][0]["why"] = "The prior is required before updating."

    compilation = module.compile_context(module.build_graph(ROOT), payload, ROOT)

    assert [edge["relation"] for edge in compilation["used_registered_edges"]] == ["competes"]
    assert compilation["unresolved_edges"][0]["relation"] == "requires"


def test_compiler_requires_evidence_connections_and_discrimination():
    module = load_graph_module()
    graph = module.build_graph(ROOT)

    payload = context_payload()
    payload["programs"] = payload["programs"][:1]
    with pytest.raises(ValueError, match="at least two competing programs"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["programs"][0]["models"][0]["evidence"] = ["obs-unknown"]
    with pytest.raises(ValueError, match="known evidence ids"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["programs"][1]["models"] = payload["programs"][1]["models"][:1]
    with pytest.raises(ValueError, match="2 to 5 models"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["programs"][0]["edges"] = []
    with pytest.raises(ValueError, match="relation graph must be connected"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["programs"][0]["prediction"]["distinguishes_from"] = "evidence-update"
    with pytest.raises(ValueError, match="distinguish another program"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["observations"][0]["status"] = "opinion"
    with pytest.raises(ValueError, match="invalid status"):
        module.compile_context(graph, payload, ROOT)


def test_compiler_rejects_unknown_duplicate_and_deprecated_models():
    module = load_graph_module()
    graph = module.build_graph(ROOT)

    payload = context_payload()
    payload["programs"][0]["models"][0]["ref"] = "mm-does-not-exist@1"
    with pytest.raises(ValueError, match="unknown"):
        module.compile_context(graph, payload, ROOT)

    payload = context_payload()
    payload["programs"][0]["models"][1] = copy.deepcopy(payload["programs"][0]["models"][0])
    with pytest.raises(ValueError, match="duplicate"):
        module.compile_context(graph, payload, ROOT)

    deprecated_graph = copy.deepcopy(graph)
    deprecated_graph["nodes"] = [
        {**node, "status": "deprecated"} if node["id"] == "mm-base-rate" else node
        for node in deprecated_graph["nodes"]
    ]
    with pytest.raises(ValueError, match="deprecated"):
        module.compile_context(deprecated_graph, context_payload(), ROOT)


def test_obsidian_index_separates_formal_and_discovery_edges():
    module = load_graph_module()
    index = module.render_obsidian_index(module.build_graph(ROOT))

    assert "Formal edges describe reusable mechanism relations" in index
    assert "`composes` (formal)" in index
    assert "`related` (discovery)" in index
    assert "proves a mechanism is active" in index


def test_mechanism_readiness_audit_reports_all_cards_ready():
    module = load_graph_module()
    graph = module.build_graph(ROOT)
    report = module.mechanism_readiness_report(graph)
    markdown = module.render_mechanism_readiness_markdown(graph)

    assert report["total_models"] == report["ready_models"] + report["legacy_models"]
    assert report["total_models"] == 88
    assert report["legacy_models"] == 0
    assert report["ready_models"] == 88
    base_rate = next(item for item in report["models"] if item["ref"] == "mm-base-rate@3")
    assert base_rate["status"] == "ready"
    assert base_rate["missing_interface_sections"] == []
    assert base_rate["empty_interface_sections"] == []
    assert "Legacy means" in markdown


def test_mechanism_readiness_rejects_empty_interface_sections():
    module = load_graph_module()
    text = """## 机制接口
### 输入
Useful input.
### 转化

### 输出
Useful output.
### 反证
Useful rival.
"""

    status = module.mechanism_interface_status(text)

    assert status["mechanism_readiness"] == "legacy"
    assert status["missing_interface_sections"] == []
    assert status["empty_interface_sections"] == ["转化"]
    assert module.interface_issue_summary(status) == "empty: 转化"


def test_context_pack_can_be_loaded_from_stdin(monkeypatch):
    module = load_graph_module()
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(json.dumps(context_payload())))

    payload = module.load_json_object(Path("-"))

    assert payload["question"]
