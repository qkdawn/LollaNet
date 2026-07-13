import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "06-Scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


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
created: 2026-07-03
updated: 2026-07-03
source_refs: []
privacy: private
---

# Accepted test proposal
""",
        encoding="utf-8",
    )
    return proposal_id


def test_create_model_writes_expected_card_and_refuses_overwrite(tmp_path):
    module = load_script("create_model.py")
    proposal_id = write_accepted_new_model_proposal(tmp_path)

    first = module.create_model(
        vault_root=tmp_path,
        name="激励机制",
        discipline="Psychology",
        importance=5,
        frequency="high",
        today="2026-07-03",
        proposal_id=proposal_id,
    )

    assert first == tmp_path / "01-Models" / "Psychology" / "激励机制.md"
    text = first.read_text(encoding="utf-8")
    assert "type: mental_model" in text
    assert "id: mm-" in text
    assert "name: 激励机制" in text
    assert "english_name: Needs English Name" in text
    assert "discipline: Psychology" in text
    assert "kind: reasoning" in text
    assert "status: candidate" in text
    assert "version: 1" in text
    assert "## 耐久机制" in text
    assert "## 时代性表现" in text
    assert "## AI 调用条件" in text
    assert "## 场景化决策提问" in text
    assert "### 职业方向" in text
    assert "### 创业/产品" in text
    assert "### 投资/商业" in text
    assert "### 合作/组织" in text
    assert "## 支持证据" in text
    assert "## 反证" in text
    assert "## 失效条件" in text
    assert "## 可观察预测" in text
    assert "## AI 输出要求" in text
    assert "## 修订历史" in text
    assert "## 使用记录" not in text
    assert "## 复盘结论" not in text
    assert "[[机会成本]]" in text
    assert "english_name: \n" not in text

    second = module.create_model(
        vault_root=tmp_path,
        name="激励机制",
        discipline="Psychology",
        importance=5,
        frequency="high",
        today="2026-07-03",
        proposal_id=proposal_id,
    )

    assert second == first
    assert first.read_text(encoding="utf-8") == text


def test_create_decision_writes_dated_open_log(tmp_path):
    module = load_script("create_decision.py")

    path = module.create_decision(
        vault_root=tmp_path,
        title="是否进入 AI 教育产品方向",
        project="test-project",
        importance="high",
        review_date="2026-08-03",
        today="2026-07-03",
        related_models=["机会成本", "能力圈", "安全边际"],
    )

    assert path == tmp_path / "private" / "projects" / "test-project" / "decisions" / "open" / "2026-07-03 是否进入 AI 教育产品方向.md"
    text = path.read_text(encoding="utf-8")
    assert "type: decision" in text
    assert "project: test-project" in text
    assert "status: open" in text
    assert "review_date: 2026-08-03" in text
    assert "- [[机会成本]]" in text
    assert "## 反向思考" in text


def test_create_decision_requires_project_unless_public_example(tmp_path):
    module = load_script("create_decision.py")

    try:
        module.create_decision(
            vault_root=tmp_path,
            title="测试决策",
            today="2026-07-03",
        )
    except ValueError as exc:
        assert "--project" in str(exc)
    else:
        raise AssertionError("create_decision should require a project for private decisions")

    try:
        module.create_decision(
            vault_root=tmp_path,
            title="公共示例",
            today="2026-07-03",
            public_example=True,
        )
    except ValueError as exc:
        assert "public examples are not supported" in str(exc)
    else:
        raise AssertionError("public examples should not be written to public decision folders")


def test_init_private_creates_private_workspace_without_overwrite(tmp_path):
    module = load_script("init_private.py")

    created = module.init_private(tmp_path)
    profile = tmp_path / "private" / "me" / "User Profile.md"
    goals = tmp_path / "private" / "me" / "Long-term Goals.md"
    principles = tmp_path / "private" / "me" / "Principles.md"
    inbox = tmp_path / "private" / "projects" / "inbox"

    assert profile in created
    assert goals in created
    assert principles in created
    assert (inbox / "Project Profile.md").exists()
    assert (inbox / "decisions" / "open").is_dir()

    profile.write_text("custom profile", encoding="utf-8")
    module.init_private(tmp_path)

    assert profile.read_text(encoding="utf-8") == "custom profile"


def test_create_project_creates_isolated_project_without_overwrite(tmp_path):
    module = load_script("create_project.py")

    project_root = module.create_project(tmp_path, "test-project", "测试项目")

    assert project_root == tmp_path / "private" / "projects" / "test-project"
    profile = project_root / "Project Profile.md"
    dashboard = project_root / "dashboard.md"
    assert profile.exists()
    assert dashboard.exists()
    assert (project_root / "decisions" / "open").is_dir()
    assert (project_root / "evidence").is_dir()
    assert "测试项目" in profile.read_text(encoding="utf-8")

    profile.write_text("custom project profile", encoding="utf-8")
    module.create_project(tmp_path, "test-project", "测试项目")

    assert profile.read_text(encoding="utf-8") == "custom project profile"


def test_validate_vault_reports_missing_type_and_broken_links(tmp_path):
    module = load_script("validate_vault.py")
    note_dir = tmp_path / "01-Models" / "Psychology"
    note_dir.mkdir(parents=True)
    (note_dir / "孤立.md").write_text("# 孤立\n\n[[不存在的模型]]\n", encoding="utf-8")

    result = module.validate_vault(tmp_path)

    assert result["error_count"] >= 2
    joined = "\n".join(result["errors"])
    assert "missing YAML frontmatter" in joined
    assert "broken wiki link" in joined


def test_validate_vault_ignores_private_by_default_and_can_include_it(tmp_path):
    module = load_script("validate_vault.py")
    for directory in module.REQUIRED_DIRS:
        (tmp_path / directory).mkdir(parents=True)
    private_dir = tmp_path / "private" / "projects" / "test-project" / "decisions" / "open"
    private_dir.mkdir(parents=True)
    (private_dir / "bad.md").write_text("# bad\n\n[[不存在]]\n", encoding="utf-8")

    public_result = module.validate_vault(tmp_path)
    private_result = module.validate_vault(tmp_path, include_private=True)

    assert public_result["error_count"] == 0
    assert private_result["error_count"] >= 2


def test_scripts_expose_help():
    for script in ["create_model.py", "create_case.py", "create_decision.py", "create_project.py", "init_private.py", "validate_vault.py"]:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "06-Scripts" / script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout.lower()


def test_all_model_cards_use_callable_ai_function_structure():
    required_sections = [
        "## 一句话定义",
        "## 核心机制",
        "## 耐久机制",
        "## 时代性表现",
        "## 适用边界",
        "## AI 调用条件",
        "## 不应调用",
        "## 场景化决策提问",
        "### 职业方向",
        "### 创业/产品",
        "### 投资/商业",
        "### 合作/组织",
        "## 典型误用",
        "## 反向思考",
        "## 经典案例",
        "### 案例事实",
        "### 模型解释",
        "### 来源",
        "## 支持证据",
        "## 反证",
        "## 失效条件",
        "## 可观察预测",
        "## 关联模型",
        "### 常一起调用",
        "### 容易混淆",
        "### 互相校正",
        "## AI 输出要求",
        "## 修订历史",
    ]
    model_paths = list((ROOT / "01-Models").rglob("*.md"))

    assert len(model_paths) == 30

    missing: list[tuple[str, str]] = []
    weak_questions: list[tuple[str, str]] = []
    for path in model_paths:
        text = path.read_text(encoding="utf-8")
        for section in required_sections:
            if section not in text:
                missing.append((path.name, section))
        for heading in ["### 职业方向", "### 创业/产品", "### 投资/商业", "### 合作/组织"]:
            if heading in text:
                block = text.split(heading, 1)[1].split("###", 1)[0].split("##", 1)[0]
                question_count = sum(1 for line in block.splitlines() if line.strip().startswith(("1.", "2.", "3.")))
                if question_count < 3:
                    weak_questions.append((path.name, heading))

    assert missing == []
    assert weak_questions == []

    for path in model_paths:
        text = path.read_text(encoding="utf-8")
        assert "## 使用记录" not in text
        assert "当前个人使用线索" not in text


def test_model_cards_do_not_keep_generic_generator_phrases():
    forbidden = [
        "芒格式用法是把概念变成检查清单",
        "用概念替代事实",
        "只在事后解释结果",
        "第一版为概念整理稿",
        "english_name: \n",
        "如果利益关系没有改善",
        "最小实验应该测哪个生存空间",
        "团队是否在用这个模型掩盖证据替代",
        "用户的问题涉及权威影响，需要判断它是否真的影响结果",
        "芒格式用法：先问",
        "如果我误用",
        "解释所有结果，忽略其他主变量",
        "看到资源匹配的表象后跳过验证",
        "属于跨学科判断工具",
    ]
    offenders: list[tuple[str, str]] = []

    for path in (ROOT / "01-Models").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            if phrase in text:
                offenders.append((path.name, phrase))

    assert offenders == []
