import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "06-Scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def frontmatter(text: str) -> str:
    assert text.startswith("---\n")
    return text.split("\n---\n", 1)[0][4:]


def scalar(block: str, key: str) -> str:
    prefix = f"{key}:"
    line = next(line for line in block.splitlines() if line.startswith(prefix))
    return line[len(prefix) :].strip()


def make_vault(tmp_path: Path, validator) -> None:
    for directory in validator.REQUIRED_DIRS:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)


def write_accepted_new_model_proposal(root: Path, proposal_id: str = "mp-test-new-model") -> str:
    proposal_dir = root / "private" / "model-proposals" / "accepted"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / f"{proposal_id}.md").write_text(
        f"""---
type: model_change_proposal
id: {proposal_id}
proposal_type: new
status: accepted
target_model_ids: []
created: 2026-07-12
updated: 2026-07-12
source_refs: []
privacy: private
---

# Accepted test proposal
""",
        encoding="utf-8",
    )
    return proposal_id


def test_all_seed_cards_use_evolvable_model_schema():
    validator = load_script("validate_vault.py")
    seed_audit = load_script("complete_seed_models.py")
    paths = sorted((ROOT / "01-Models").rglob("*.md"))

    assert len(paths) >= 30
    assert seed_audit.audit_seed_models(ROOT) == []
    ids: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        block = frontmatter(text)
        model_id = scalar(block, "id")

        assert validator.MODEL_ID_RE.fullmatch(model_id)
        assert model_id not in ids
        ids.add(model_id)
        rel = path.relative_to(ROOT / "01-Models")
        assert len(rel.parts) == 3
        assert scalar(block, "source_category") in validator.SOURCE_CATEGORIES
        assert scalar(block, "discipline")
        assert scalar(block, "subdiscipline")
        assert scalar(block, "formalism") in {"true", "false"}
        assert rel.parts[0] == scalar(block, "source_category")
        assert rel.parts[1] == scalar(block, "discipline")
        assert scalar(block, "kind") in validator.MODEL_KINDS
        assert scalar(block, "status") in validator.MODEL_STATUSES
        assert int(scalar(block, "version")) >= 1
        assert "conflicts:" in block
        assert "supersedes:" in block
        for section in validator.MODEL_SECTIONS:
            assert section in text, (path.name, section)
        assert "## 使用记录" not in text
        assert "当前个人使用线索" not in text


def test_math_candidates_reference_accepted_private_batch_proposal():
    proposal_id = "mp-2026-07-15-new-mathematics-foundations"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: new" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    math_cards = sorted((ROOT / "01-Models" / "学科知识" / "数学").glob("*.md"))
    assert len(math_cards) == 12
    for path in math_cards:
        text = path.read_text(encoding="utf-8")
        assert "status: candidate" in text
        assert proposal_id in text, path.name


def test_physics_candidates_reference_accepted_private_batch_proposal():
    proposal_id = "mp-2026-07-15-new-physics-foundations"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: new" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    physics_names = {
        "量纲分析与尺度律",
        "力与平衡",
        "动量与冲量",
        "能量守恒与功",
        "熵与不可逆性",
        "扩散与梯度驱动",
        "振荡与共振",
        "波动与传播",
        "场与势",
        "稳定性与相变",
    }
    physics_cards = sorted(
        (ROOT / "01-Models" / "学科知识" / "物理学" / name).with_suffix(".md")
        for name in physics_names
    )
    assert all(path.exists() for path in physics_cards)
    for path in physics_cards:
        text = path.read_text(encoding="utf-8")
        assert "discipline: 物理学" in text
        assert "formalism: true" in text
        assert "status: candidate" in text
        assert proposal_id in text, path.name


def test_quantum_candidates_reference_accepted_private_batch_proposal():
    proposal_id = "mp-2026-07-15-new-quantum-foundations"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: new" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    quantum_names = {
        "状态向量与叠加",
        "哈密顿量与时间演化",
        "量子化与离散能级",
        "测量与 Born 规则",
        "不确定性关系",
        "纠缠与 Bell 相关",
        "隧穿与势垒",
        "退相干与经典涌现",
        "费米玻色统计与泡利不相容",
    }
    quantum_cards = sorted(
        (ROOT / "01-Models" / "学科知识" / "物理学" / name).with_suffix(".md")
        for name in quantum_names
    )
    assert all(path.exists() for path in quantum_cards)
    for path in quantum_cards:
        text = path.read_text(encoding="utf-8")
        assert "discipline: 物理学" in text
        assert "formalism: true" in text
        assert "status: candidate" in text
        assert proposal_id in text, path.name



def test_chemistry_candidates_reference_accepted_private_batch_proposal():
    proposal_id = "mp-2026-07-16-new-chemistry-foundations"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: new" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    chemistry_names = {
        "物质守恒、化学计量与限制试剂",
        "原子结构与周期趋势",
        "化学键、分子几何与极性",
        "分子间作用力与相态",
        "反应焓与 Hess 路径",
        "Gibbs 自由能、化学势与自发性",
        "速率律与控速步骤",
        "活化能、催化与替代路径",
        "反应商、平衡常数与扰动响应",
        "酸碱平衡、缓冲与容量",
        "氧化还原、电势与电子转移",
        "溶解、分配与选择性",
    }
    chemistry_cards = sorted(
        (ROOT / "01-Models" / "学科知识" / "化学" / name).with_suffix(".md")
        for name in chemistry_names
    )
    assert len(chemistry_cards) == 12
    assert all(path.exists() for path in chemistry_cards)
    for path in chemistry_cards:
        text = path.read_text(encoding="utf-8")
        assert "discipline: 化学" in text
        assert "formalism: true" in text
        assert "status: candidate" in text
        assert proposal_id in text, path.name



def test_engineering_candidates_reference_accepted_private_batch_proposal():
    proposal_id = "mp-2026-07-16-new-engineering-foundations"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: new" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    engineering_names = {
        "需求、约束与可验证指标",
        "系统边界、运行环境与利益相关者",
        "功能分解、需求分配与层级",
        "接口、耦合与集成顺序",
        "权衡空间、Pareto 前沿与决策记录",
        "误差预算、不确定性传播与容差链",
        "鲁棒设计、敏感度与最坏情形",
        "失效模式、故障树与危害路径",
        "可靠度、失效率与寿命分布",
        "可维护性、可诊断性与恢复时间",
        "验证、确认与双向可追溯性",
        "生命周期成本、技术债与可演化性",
    }
    engineering_cards = sorted(
        (ROOT / "01-Models" / "学科知识" / "工程学" / name).with_suffix(".md")
        for name in engineering_names
    )
    assert len(engineering_cards) == 12
    assert all(path.exists() for path in engineering_cards)
    for path in engineering_cards:
        text = path.read_text(encoding="utf-8")
        assert "discipline: 工程学" in text
        assert "formalism: true" in text
        assert "status: candidate" in text
        assert proposal_id in text, path.name



def test_engineering_core_formalism_amendment_is_accepted():
    proposal_id = "mp-2026-07-16-amend-engineering-core-formalism"
    proposal_path = ROOT / "private" / "model-proposals" / "accepted" / f"{proposal_id}.md"
    proposal_text = proposal_path.read_text(encoding="utf-8")

    assert "type: model_change_proposal" in proposal_text
    assert f"id: {proposal_id}" in proposal_text
    assert "proposal_type: amend" in proposal_text
    assert "status: accepted" in proposal_text
    assert "privacy: private" in proposal_text

    engineering_core = {
        "安全边际": "mm-margin-of-safety",
        "冗余": "mm-redundancy",
        "瓶颈": "mm-bottleneck",
        "反馈回路": "mm-feedback-loop",
    }
    formal_sections = {
        "## 形式化内容",
        "### 定义与符号",
        "### 公式或算法",
        "### 成立前提",
        "### 推导或证明来源",
        "### 反例与边界",
        "### 跨域转译",
    }
    for name, model_id in engineering_core.items():
        assert model_id in proposal_text
        card_path = ROOT / "01-Models" / "学科知识" / "工程学" / f"{name}.md"
        text = card_path.read_text(encoding="utf-8")
        assert f"id: {model_id}" in text
        assert "discipline: 工程学" in text
        assert "formalism: true" in text
        assert "status: active" in text
        assert "version: 3" in text
        assert proposal_id in text
        assert all(section in text for section in formal_sections)


def test_model_evolution_phase_zero_protocol_contract_is_opt_in_and_manual():
    protocol_path = ROOT / "04-Prompts" / "Model Evolution Protocol.md"
    template_path = ROOT / "03-Templates" / "model_usage_evidence_template.md"
    checklist_path = ROOT / "02-Checklists" / "Model Portfolio Review Checklist.md"
    analysis_path = ROOT / "04-Prompts" / "LollaNet Analysis Protocol.md"
    spec_path = ROOT / "PROJECT_SPEC.md"
    readme_path = ROOT / "README.md"
    external_plan = ROOT / "07-Docs" / "plans" / "external-input-loop.md"
    personalized_plan = ROOT / "07-Docs" / "plans" / "personalized-model-evolution-loop.md"

    assert protocol_path.exists()
    assert template_path.exists()
    assert checklist_path.exists()

    protocol = protocol_path.read_text(encoding="utf-8")
    template = template_path.read_text(encoding="utf-8")
    checklist = checklist_path.read_text(encoding="utf-8")
    analysis = analysis_path.read_text(encoding="utf-8")
    spec = spec_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")

    for required in [
        "type: prompt",
        "status: active",
        "用户明确要求**保存、复盘或检验**",
        "默认不保存",
        "外部公开输入",
        "真实任务与私人适配",
        "private/evaluations/",
        "private/projects/<project_id>/reviews/",
        "公开、独立、可核验",
        "30 天",
        "90 天",
        "调用次数、成功率或固定加权总分",
        "不得自动",
        "不沉淀",
        "展示 proposal 草案",
    ]:
        assert required in protocol

    for required in [
        "模型调用与独立行动增量",
        "如果不调用该模型，结论会如何不同",
        "获授权的私人上下文",
        "未使用的相关私人资料类别",
        "最强反证、适用边界或未知量",
        "30 天复盘",
        "90 天复盘",
        "建议处置",
        "不要回写公共模型卡",
    ]:
        assert required in template

    for required in [
        "独立行动增量",
        "预测区分度",
        "重复",
        "冲突",
        "低频但覆盖罕见高损失风险",
        "私人适配与公共模型边界",
        "不得直接改变模型状态",
    ]:
        assert required in checklist

    assert "[Model Evolution Protocol](Model%20Evolution%20Protocol.md)" in analysis
    assert "默认不保存任务记录" in analysis
    assert "不得为此额外读取私人资料" in analysis
    assert "不得自动创建 proposal 或修改任何正式模型" in analysis
    assert "`04-Prompts/Model Evolution Protocol.md`" in spec
    assert "[Model Evolution Protocol](04-Prompts/Model%20Evolution%20Protocol.md)" in readme
    assert "`04-Prompts/Model Evolution Protocol.md`" in external_plan.read_text(encoding="utf-8")
    assert "`04-Prompts/Model Evolution Protocol.md`" in personalized_plan.read_text(encoding="utf-8")


def test_model_test_v2_contract_separates_system_ablation_and_governance():
    protocol_path = ROOT / "07-Docs" / "model-test-protocol-v2.md"
    charter_path = ROOT / "03-Templates" / "model_test_charter_template.md"
    blind_path = ROOT / "03-Templates" / "blind_evaluation_v2_record_template.md"
    ablation_path = ROOT / "03-Templates" / "model_ablation_record_template.md"
    matrix_path = ROOT / "08-Examples" / "evaluation" / "benchmark-v2-task-matrix.md"
    engineering_path = ROOT / "08-Examples" / "evaluation" / "engineering-core-ablation-cases-v2.md"
    readme_path = ROOT / "README.md"
    spec_path = ROOT / "PROJECT_SPEC.md"

    for path in [protocol_path, charter_path, blind_path, ablation_path, matrix_path, engineering_path]:
        assert path.exists()

    protocol = protocol_path.read_text(encoding="utf-8")
    charter = charter_path.read_text(encoding="utf-8")
    blind = blind_path.read_text(encoding="utf-8")
    ablation = ablation_path.read_text(encoding="utf-8")
    matrix = matrix_path.read_text(encoding="utf-8")
    engineering = engineering_path.read_text(encoding="utf-8")

    for required in [
        "卡片质量",
        "系统增量",
        "模型独立增量",
        "长期适用性",
        "Baseline / Full / Ablation",
        "24 个完全虚构任务",
        "调用次数、短期成功率或固定加权总分",
        "不得自动改变模型状态",
        "公开、独立、可核验",
        "30 天",
        "90 天",
    ]:
        assert required in protocol

    for required in [
        "在生成任何答案前冻结本页",
        "不得读取、披露或保存的私人上下文",
        "全量报告",
        "自动创建 proposal 或修改模型",
    ]:
        assert required in charter

    for required in [
        "相同任务文字、外部证据、工具权限和答案预算",
        "Full 实际选择的模型",
        "没有读取或泄露无关私人资料",
        "不直接证明某张单独模型卡有效",
    ]:
        assert required in blind

    for required in [
        "目标模型 ID",
        "Ablation 模型集合（必须等于 Full 减去目标模型）",
        "不得以提示词、换名或新模型重建目标机制",
        "独立行动增量",
        "不授权修改模型",
    ]:
        assert required in ablation

    for required in [
        "fictional: true",
        "case_count: 24",
        "S01 / A01",
        "S12 / A12",
        "不得把“预期检验”照抄为答案",
    ]:
        assert required in matrix

    for required in [
        "mm-margin-of-safety",
        "mm-redundancy",
        "mm-bottleneck",
        "mm-feedback-loop",
        "每张工程模型在三个不同任务中测试",
        "自动升级、自动合并、自动淘汰",
    ]:
        assert required in engineering

    assert "[Model Test Protocol v2](07-Docs/model-test-protocol-v2.md)" in readme_path.read_text(encoding="utf-8")
    assert "`07-Docs/model-test-protocol-v2.md`" in spec_path.read_text(encoding="utf-8")


def test_public_model_library_has_no_quality_debt():
    validator = load_script("validate_vault.py")

    result = validator.validate_vault(ROOT)
    quality_warnings = [warning for warning in result["warnings"] if "model quality debt" in warning]

    assert result["errors"] == []
    assert quality_warnings == []


def test_materially_disputed_models_are_contested():
    expected = {
        "mm-authority-bias": "contested",
        "mm-loss-aversion": "contested",
    }
    observed: dict[str, str] = {}

    for path in (ROOT / "01-Models").rglob("*.md"):
        block = frontmatter(path.read_text(encoding="utf-8"))
        model_id = scalar(block, "id")
        if model_id in expected:
            observed[model_id] = scalar(block, "status")

    assert observed == expected


def test_create_model_writes_candidate_with_stable_id_and_no_public_usage_log(tmp_path):
    module = load_script("create_model.py")
    proposal_id = write_accepted_new_model_proposal(tmp_path)

    path = module.create_model(
        tmp_path,
        "测试模型",
        discipline="Engineering",
        kind="action",
        today="2026-07-12",
        proposal_id=proposal_id,
    )
    text = path.read_text(encoding="utf-8")
    model_id = module.default_model_id("测试模型", "Engineering")

    assert f"id: {model_id}" in text
    assert "kind: action" in text
    assert "status: candidate" in text
    assert "version: 1" in text
    assert "conflicts: []" in text
    assert "supersedes: []" in text
    assert "## 可观察预测" in text
    assert "## 修订历史" in text
    assert "## 使用记录" not in text
    assert module.default_model_id("测试模型", "Engineering") == model_id

    with pytest.raises(ValueError, match="kind"):
        module.model_body("坏模型", "数学", 3, "low", "2026-07-12", kind="unknown")
    with pytest.raises(ValueError, match="model_id"):
        module.model_body("坏模型", "数学", 3, "low", "2026-07-12", model_id="not stable")


def test_create_model_requires_an_accepted_new_model_proposal(tmp_path):
    module = load_script("create_model.py")

    with pytest.raises(ValueError, match="requires --proposal"):
        module.create_model(tmp_path, "Unreviewed model", today="2026-07-12")

    proposed_dir = tmp_path / "private" / "model-proposals" / "accepted"
    proposed_dir.mkdir(parents=True)
    (proposed_dir / "mp-not-accepted.md").write_text(
        """---
type: model_change_proposal
id: mp-not-accepted
proposal_type: new
status: proposed
target_model_ids: []
created: 2026-07-12
updated: 2026-07-12
source_refs: []
privacy: private
---
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is not accepted"):
        module.create_model(
            tmp_path,
            "Not accepted model",
            today="2026-07-12",
            proposal_id="mp-not-accepted",
        )


def test_create_model_refuses_duplicate_stable_id(tmp_path):
    module = load_script("create_model.py")
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    module.create_model(
        tmp_path,
        "First model",
        model_id="mm-shared-id",
        proposal_id=proposal_id,
        today="2026-07-12",
    )

    with pytest.raises(ValueError, match="proposal already created a model"):
        module.create_model(
            tmp_path,
            "Reused proposal",
            model_id="mm-different-id",
            proposal_id=proposal_id,
            today="2026-07-12",
        )

    second_proposal_id = write_accepted_new_model_proposal(tmp_path, "mp-second-new-model")
    with pytest.raises(ValueError, match="model_id already exists"):
        module.create_model(
            tmp_path,
            "Second model",
            model_id="mm-shared-id",
            proposal_id=second_proposal_id,
            today="2026-07-12",
        )


def test_validator_rejects_invalid_evolution_metadata(tmp_path):
    validator = load_script("validate_vault.py")
    creator = load_script("create_model.py")
    make_vault(tmp_path, validator)
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    path = creator.create_model(
        tmp_path,
        "测试模型",
        discipline="数学",
        subdiscipline="测试",
        kind="world",
        today="2026-07-12",
        proposal_id=proposal_id,
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace("kind: world", "kind: slogan")
    text = text.replace("status: candidate", "status: seed")
    text = text.replace("version: 1", "version: 0", 1)
    path.write_text(text, encoding="utf-8")

    result = validator.validate_vault(tmp_path)
    joined = "\n".join(result["errors"])

    assert "invalid model kind 'slogan'" in joined
    assert "invalid model status 'seed'" in joined
    assert "model version must be a positive integer" in joined


def test_create_model_writes_canonical_taxonomy_metadata(tmp_path):
    creator = load_script("create_model.py")
    proposal_id = write_accepted_new_model_proposal(tmp_path)

    path = creator.create_model(
        tmp_path,
        "形式化测试模型",
        source_category="学科知识",
        discipline="数学",
        subdiscipline="微积分",
        formalism=True,
        kind="reasoning",
        proposal_id=proposal_id,
        today="2026-07-12",
    )

    assert path == tmp_path / "01-Models" / "学科知识" / "数学" / "形式化测试模型.md"
    text = path.read_text(encoding="utf-8")
    assert "source_category: 学科知识" in text
    assert "discipline: 数学" in text
    assert "subdiscipline: 微积分" in text
    assert "formalism: true" in text


def test_validator_rejects_taxonomy_path_mismatch(tmp_path):
    validator = load_script("validate_vault.py")
    creator = load_script("create_model.py")
    make_vault(tmp_path, validator)
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    path = creator.create_model(
        tmp_path,
        "路径测试模型",
        discipline="数学",
        subdiscipline="离散数学",
        proposal_id=proposal_id,
        today="2026-07-12",
    )
    wrong_path = tmp_path / "01-Models" / "学科知识" / "物理学" / path.name
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    path.replace(wrong_path)

    result = validator.validate_vault(tmp_path)

    assert "model path does not match taxonomy" in "\n".join(result["errors"])


def test_validator_rejects_incomplete_formal_model(tmp_path):
    validator = load_script("validate_vault.py")
    creator = load_script("create_model.py")
    make_vault(tmp_path, validator)
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    creator.create_model(
        tmp_path,
        "不完整形式化模型",
        source_category="学科知识",
        discipline="数学",
        subdiscipline="测试",
        formalism=True,
        proposal_id=proposal_id,
        today="2026-07-12",
    )

    result = validator.validate_vault(tmp_path)
    errors = "\n".join(result["errors"])

    assert "formal model missing required section" in errors


def test_validator_warns_when_active_model_still_has_seed_quality_debt(tmp_path):
    validator = load_script("validate_vault.py")
    creator = load_script("create_model.py")
    make_vault(tmp_path, validator)
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    path = creator.create_model(
        tmp_path,
        "测试模型",
        discipline="数学",
        subdiscipline="测试",
        kind="world",
        today="2026-07-12",
        proposal_id=proposal_id,
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace("status: candidate", "status: active")
    text = text.replace("来源需要核验。这里只能补充可操作转述，不得伪造芒格原话。", "这个案例用于检查：测试机制。")
    path.write_text(text, encoding="utf-8")

    result = validator.validate_vault(tmp_path)
    joined = "\n".join(result["warnings"])

    assert "active model quality debt" in joined
    assert "no attributable source in case or supporting evidence" in joined
    assert "no attributable counterevidence or boundary source" in joined
    assert "seed boilerplate phrase" in joined


def test_validator_accepts_grounded_active_model_without_quality_warning(tmp_path):
    validator = load_script("validate_vault.py")
    creator = load_script("create_model.py")
    make_vault(tmp_path, validator)
    proposal_id = write_accepted_new_model_proposal(tmp_path)
    path = creator.create_model(
        tmp_path,
        "测试模型",
        discipline="数学",
        subdiscipline="测试",
        kind="world",
        today="2026-07-12",
        proposal_id=proposal_id,
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace("status: candidate", "status: active")
    text = text.replace(
        "- 核心来源：待验证线索；接受前替换为 `[作者或机构，标题（年份）](可回引 URL 或 DOI)`。",
        "- 核心来源：[支持研究](https://doi.org/10.1000/support)。",
    )
    text = text.replace(
        "- 最强反证或边界来源：待验证线索。",
        "- 最强反证或边界来源：[边界研究](https://doi.org/10.1000/boundary)。",
    )
    path.write_text(text, encoding="utf-8")

    result = validator.validate_vault(tmp_path)

    assert not any("active model quality debt" in warning for warning in result["warnings"])


def test_validator_enforces_private_model_change_proposals(tmp_path):
    validator = load_script("validate_vault.py")
    make_vault(tmp_path, validator)
    proposal = """---
type: model_change_proposal
id: mp-2026-07-12-test
proposal_type: amend
status: proposed
target_model_ids: []
created: 2026-07-12
updated: 2026-07-12
source_refs: []
privacy: public
---

# Test proposal
"""
    public_path = tmp_path / "08-Examples" / "proposal.md"
    public_path.write_text(proposal, encoding="utf-8")

    public_result = validator.validate_vault(tmp_path)
    public_errors = "\n".join(public_result["errors"])
    assert "model-change proposals may only exist under private/" in public_errors
    assert "model-change proposal privacy must be private" in public_errors
    assert "non-new model-change proposals require target_model_ids" in public_errors

    public_path.unlink()
    private_dir = tmp_path / "private" / "model-proposals" / "proposed"
    private_dir.mkdir(parents=True)
    (private_dir / "proposal.md").write_text(proposal.replace("privacy: public", "privacy: private"), encoding="utf-8")
    private_result = validator.validate_vault(tmp_path, include_private=True)
    private_errors = "\n".join(private_result["errors"])
    assert "may only exist under private/" not in private_errors
    assert "non-new model-change proposals require target_model_ids" in private_errors


def test_validator_requires_proposal_status_directory_to_match(tmp_path):
    validator = load_script("validate_vault.py")
    make_vault(tmp_path, validator)
    proposal_dir = tmp_path / "private" / "model-proposals" / "accepted"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.md").write_text(
        """---
type: model_change_proposal
id: mp-test-new-model
proposal_type: new
status: proposed
target_model_ids: []
created: 2026-07-12
updated: 2026-07-12
source_refs: []
privacy: private
---

# Test
""",
        encoding="utf-8",
    )

    result = validator.validate_vault(tmp_path, include_private=True)
    assert "must match directory 'accepted'" in "\n".join(result["errors"])


def test_retired_seed_writer_is_read_only():
    model_paths = sorted((ROOT / "01-Models").rglob("*.md"))
    before = {path: path.read_bytes() for path in model_paths}

    completed = subprocess.run(
        [sys.executable, str(ROOT / "06-Scripts" / "complete_seed_models.py"), "--root", str(ROOT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "no files were changed" in completed.stdout
    assert {path: path.read_bytes() for path in model_paths} == before


def test_seed_audit_allows_new_accepted_models(tmp_path):
    module = load_script("complete_seed_models.py")
    model_root = tmp_path / "01-Models" / "Test"
    model_root.mkdir(parents=True)
    for name in module.EXPECTED_MODEL_NAMES | {"未来新增模型"}:
        (model_root / f"{name}.md").write_text("placeholder", encoding="utf-8")

    assert module.audit_seed_models(tmp_path) == []
