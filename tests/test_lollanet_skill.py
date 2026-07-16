from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".agents" / "skills" / "lollanet-analyze"
SKILL_PATH = SKILL_DIR / "SKILL.md"
OPENAI_PATH = SKILL_DIR / "agents" / "openai.yaml"
PROTOCOL_PATH = ROOT / "04-Prompts" / "LollaNet Analysis Protocol.md"


def read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    frontmatter, _ = text[4:].split("\n---\n", 1)
    values: dict[str, str] = {}
    for line in frontmatter.splitlines():
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def test_skill_metadata_supports_explicit_and_implicit_invocation():
    skill = read_skill()
    metadata = parse_frontmatter(skill)
    openai = OPENAI_PATH.read_text(encoding="utf-8")

    assert metadata["name"] == "lollanet-analyze"
    description = metadata["description"].lower()
    assert "$lollanet-analyze" in description
    assert "implicitly" in description
    assert "simple factual" in description
    assert 'display_name: "LollaNet Analyze"' in openai
    assert "$lollanet-analyze" in openai
    assert "allow_implicit_invocation: true" in openai
    assert "TODO" not in skill


def test_skill_uses_protocol_as_source_and_handles_execution_modes():
    skill = read_skill()

    assert "04-Prompts/LollaNet Analysis Protocol.md" in skill
    assert "sole source of truth" in skill
    assert "skip all scouts" in skill
    assert "no more than three read-only scouts" in skill
    assert "model-scout" in skill
    assert "memory-scout" in skill
    assert "challenge-scout" in skill
    assert "perform the same model, memory-if-needed, and challenge searches sequentially" in skill
    assert "vector database" in skill
    assert "09-Cases/" in skill
    assert "source_refs" in skill


def test_skill_defines_complete_retrieval_and_synthesis_contract():
    skill = read_skill()
    packet_fields = [
        "kind:",
        "title:",
        "location:",
        "source_category:",
        "match_signal:",
        "basis:",
        "boundary_or_counterevidence:",
        "as_of:",
        "confidence:",
    ]

    for field in packet_fields:
        assert field in skill
    assert "public-model | private-model" in skill
    assert "private-value" in skill
    assert "private-goal" in skill
    assert "private-focus" in skill
    assert "scout's conclusion is never evidence by itself" in skill
    assert "narrowest underlying heading" in skill
    assert "document-title anchor is insufficient" in skill
    assert "Reject unsupported items" in skill
    assert "select 3-7 non-redundant models" in skill
    assert "never pad the set" in skill
    assert "majority vote" in skill
    assert "metadata conflicts and incompatible predictions" in skill


def test_skill_keeps_simple_answers_clean_and_unknown_gates_reversible():
    skill = read_skill()

    assert "do not expose routing commentary" in skill
    assert "distinguish a low-cost information-gathering step from substantive commitment" in skill
    assert "minimum bounded effort needed to verify that gate" in skill
    assert "Allocate zero substantive resources" in skill
    assert "merely for symmetry" in skill
    assert "Never turn missing information into an arbitrary allocation split" in skill


def test_skill_runs_munger_dual_track_as_two_passes_without_voting():
    skill = read_skill()

    assert "Munger-style dual-track analysis" in skill
    assert "rational analysis track" in skill
    assert "psychological misjudgment audit track" in skill
    assert "two passes, not two scouts or votes" in skill
    assert "Bias risk is not proof that a conclusion is false" in skill
    assert "rerun the affected rational step" in skill


def test_protocol_defines_both_tracks_and_their_reconciliation():
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert "## 8. 芒格式双轨分析" in protocol
    assert "### 8.1 理性分析轨" in protocol
    assert "### 8.2 心理误判审计轨" in protocol
    assert "### 8.3 两轨合流" in protocol
    assert "不是两个 Scout、两个独立 Agent 或两票表决" in protocol
    assert "心理风险不是结论错误的证据" in protocol
    assert "Agent 不能直接观察人的潜意识" in protocol


def test_skill_minimizes_private_data_and_requires_write_approval():
    skill = read_skill()

    assert "read the smallest relevant sections" in skill
    assert "Do not send private content to web search" in skill
    assert "Never write real user data to public framework folders" in skill
    assert "read-only by default" in skill
    assert "An analysis request or skill invocation is not approval" in skill
    assert "after the user explicitly approves the displayed proposal" in skill
    assert "Never directly rewrite a formal model from this skill" in skill
    assert "never infer a terminal value from behavior" in skill
    assert "cannot silently rewrite them or create permanent global weights" in skill


def test_protocol_and_skill_support_divergent_then_convergent_exploration():
    skill = read_skill()
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    checklist = (ROOT / "02-Checklists" / "Analysis Quality Checklist.md").read_text(encoding="utf-8")
    template_path = ROOT / "03-Templates" / "divergent_project_framing_template.md"
    contract_path = ROOT / "07-Docs" / "exploratory-analysis-test-contract.md"

    assert "### 1.1 分析模式" in protocol
    assert "发散—收敛探索模式" in protocol
    assert "8–12 张卡" in protocol
    assert "至少四类不同机制" in protocol
    assert "3–5 个彼此竞争的项目、产品、研究或媒介命题" in protocol
    assert "不得把单一技术架构、MVP、执行路线或产品定位伪装成答案" in protocol
    assert "近期相关分析中重复调用同一模型" in protocol

    assert "exploratory-to-convergent" in skill
    assert "8-12-card candidate pool" in skill
    assert "at least four distinct mechanism families" in skill
    assert "3-5 genuinely competing project, product, research, or media framings" in skill
    assert "Do not present a single technical architecture, MVP, execution plan, or product positioning" in skill
    assert "new task-specific delta" in skill

    assert template_path.exists()
    template = template_path.read_text(encoding="utf-8")
    for required in [
        "默认不保存实例",
        "8–12 个候选模型或机制",
        "3–5 个彼此不可互换的命题",
        "为什么该媒介不可替代",
        "最小可证伪信号",
        "最大失败模式",
        "本次相对近期调用的新增量",
    ]:
        assert required in template

    assert "## 问题定义与分析模式" in checklist
    assert "彼此竞争、不可互换的命题" in checklist
    assert "重复调用的模型是否说明了本次新的行动增量" in checklist

    assert contract_path.exists()
    contract = contract_path.read_text(encoding="utf-8")
    for required in [
        "fictional: true",
        "发散—收敛探索模式",
        "至少三个彼此竞争的命题",
        "XR 为什么不可替代",
        "直接给出单一技术架构、完整 MVP 或八周路线图",
        "不证明任何模型、产品命题、XR 机制或用户体验有效",
    ]:
        assert required in contract
