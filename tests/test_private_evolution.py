import importlib.util
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


def test_init_private_adds_cognitive_assets_without_overwrite(tmp_path):
    module = load_script("init_private.py")

    created = module.init_private(tmp_path, today="2026-07-12")
    principles = tmp_path / "private" / "me" / "Principles.md"
    value_function = tmp_path / "private" / "me" / "Value Function.md"
    current_focus = tmp_path / "private" / "me" / "Current Focus.md"

    assert principles in created
    assert value_function in created
    assert current_focus in created
    assert "type: personal_principles" in principles.read_text(encoding="utf-8")
    value_text = value_function.read_text(encoding="utf-8")
    assert "type: personal_value_function" in value_text
    assert "status: draft" in value_text
    assert "Global numeric weights are" in value_text
    assert "### vf-example" in value_text
    assert "type: current_focus" in current_focus.read_text(encoding="utf-8")
    assert (tmp_path / "private" / "me" / "observations").is_dir()
    assert (tmp_path / "private" / "me" / "reviews").is_dir()
    assert (tmp_path / "private" / "me" / "proposals" / "proposed").is_dir()
    assert (tmp_path / "private" / "models").is_dir()
    for status in ["proposed", "accepted", "rejected", "deferred"]:
        assert (tmp_path / "private" / "model-proposals" / status).is_dir()
    assert (tmp_path / "private" / "evaluations").is_dir()
    assert (tmp_path / "private" / "evaluations" / "episodes").is_dir()

    principles.write_text("my private principles", encoding="utf-8")
    value_function.write_text("my private values", encoding="utf-8")
    module.init_private(tmp_path, today="2026-07-13")
    assert principles.read_text(encoding="utf-8") == "my private principles"
    assert value_function.read_text(encoding="utf-8") == "my private values"


def test_public_personal_templates_keep_asset_types_distinct():
    templates = ROOT / "03-Templates"
    profile = (templates / "user_profile_template.md").read_text(encoding="utf-8")
    values = (templates / "value_function_template.md").read_text(encoding="utf-8")
    principles = (templates / "principles_template.md").read_text(encoding="utf-8")
    goals = (templates / "long_term_goals_template.md").read_text(encoding="utf-8")
    focus = (templates / "current_focus_template.md").read_text(encoding="utf-8")

    assert "factual identity" in profile
    assert "stable `vf-*` IDs" in values
    assert "Global numeric weights are forbidden" in values
    assert "Protects values: `vf-*`" in principles
    assert "Serves values: `vf-*`" in goals
    assert "6-12 week allocation" in focus


def test_create_case_requires_an_explicit_private_or_public_destination(tmp_path):
    module = load_script("create_case.py")

    with pytest.raises(ValueError, match="exactly one"):
        module.create_case(tmp_path, "Unscoped case", today="2026-07-12")
    with pytest.raises(ValueError, match="exactly one"):
        module.create_case(
            tmp_path,
            "Ambiguous case",
            project="inbox",
            public_example=True,
            today="2026-07-12",
        )
    with pytest.raises(ValueError, match="exactly one"):
        module.create_case(
            tmp_path,
            "Ambiguous public case",
            public_case=True,
            public_example=True,
            today="2026-07-12",
        )


def test_create_case_keeps_real_material_private(tmp_path):
    module = load_script("create_case.py")

    path = module.create_case(
        tmp_path,
        "Failed rollout",
        domain="Historical",
        project="research",
        sources=["https://example.com/public-background"],
        today="2026-07-12",
    )

    assert path == (
        tmp_path
        / "private"
        / "projects"
        / "research"
        / "evidence"
        / "cases"
        / "2026-07-12 Failed rollout.md"
    )
    text = path.read_text(encoding="utf-8")
    assert "privacy: private" in text
    assert "provenance: private-material" in text
    assert "project: research" in text
    assert "date_observed: unknown" in text
    assert '  - "https://example.com/public-background"' in text
    assert "## 时代与制度条件" in text
    assert "## 事实、推断与待验证项" in text
    assert "## 竞争性解释与反证" in text


def test_create_case_writes_wholly_fictional_public_examples(tmp_path):
    module = load_script("create_case.py")

    path = module.create_case(
        tmp_path,
        "Fictional incentive case",
        public_example=True,
        today="2026-07-12",
    )

    assert path == tmp_path / "08-Examples" / "cases" / "2026-07-12 Fictional incentive case.md"
    text = path.read_text(encoding="utf-8")
    assert "privacy: public" in text
    assert "provenance: fictional" in text
    assert "example: true" in text
    assert "fictional: true" in text

    with pytest.raises(ValueError, match="must not claim real source"):
        module.create_case(
            tmp_path,
            "Confused example",
            public_example=True,
            sources=["https://example.com/real-source"],
            today="2026-07-12",
        )

def test_create_case_writes_sourced_public_real_cases(tmp_path):
    module = load_script("create_case.py")

    with pytest.raises(ValueError, match="at least one --source"):
        module.create_case(
            tmp_path,
            "Unsourced public case",
            public_case=True,
            today="2026-07-12",
        )

    path = module.create_case(
        tmp_path,
        "Published product change",
        domain="Technology",
        public_case=True,
        sources=["https://example.com/company-report (accessed 2026-07-12)"],
        observed_on="2026-06-30",
        as_of="2026-07-11",
        today="2026-07-12",
    )

    assert path == tmp_path / "09-Cases" / "Technology" / "2026-07-12 Published product change.md"
    text = path.read_text(encoding="utf-8")
    assert "privacy: public" in text
    assert "provenance: public-sourced" in text
    assert "verification_status: sourced" in text
    assert "public_case: true" in text
    assert "created: 2026-07-12" in text
    assert "updated: 2026-07-12" in text
    assert "date_observed: 2026-06-30" in text
    assert "as_of: 2026-07-11" in text
    assert '  - "https://example.com/company-report (accessed 2026-07-12)"' in text
    assert "fictional: true" not in text

    public_person_path = module.create_case(
        tmp_path,
        "Published biography",
        domain="Personal",
        public_case=True,
        sources=["ISBN 978-0-00-000000-0"],
        today="2026-07-12",
    )
    assert public_person_path.parent == tmp_path / "09-Cases" / "Personal"


def test_create_case_rejects_path_escaping_inputs(tmp_path):
    module = load_script("create_case.py")

    with pytest.raises(ValueError, match="unsafe in a filename"):
        module.create_case(tmp_path, "../outside", project="inbox", today="2026-07-12")
    with pytest.raises(ValueError, match="project must use"):
        module.create_case(tmp_path, "Case", project="../outside", today="2026-07-12")
    with pytest.raises(ValueError, match="as_of must use"):
        module.create_case(
            tmp_path,
            "Bad date",
            public_case=True,
            sources=["DOI:10.0000/example"],
            as_of="someday",
            today="2026-07-12",
        )
    with pytest.raises(ValueError, match="single-line"):
        module.create_case(
            tmp_path,
            "Bad source",
            public_case=True,
            sources=["https://example.com\n## injected"],
            today="2026-07-12",
        )


def test_validator_accepts_sourced_public_case(tmp_path):
    creator = load_script("create_case.py")
    validator = load_script("validate_vault.py")
    for directory in validator.REQUIRED_DIRS:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    creator.create_case(
        tmp_path,
        "Public evidence",
        public_case=True,
        sources=["DOI:10.0000/example"],
        as_of="2026-07-12",
        today="2026-07-12",
    )
    result = validator.validate_vault(tmp_path)

    assert result["error_count"] == 0, result["errors"]


def test_validator_rejects_unsourced_misplaced_or_badly_dated_public_cases(tmp_path):
    creator = load_script("create_case.py")
    validator = load_script("validate_vault.py")
    for directory in validator.REQUIRED_DIRS:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    path = creator.create_case(
        tmp_path,
        "Invalid public evidence",
        public_case=True,
        sources=["DOI:10.0000/example"],
        today="2026-07-12",
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace('source_refs: \n  - "DOI:10.0000/example"\n', "source_refs: []\n")
    text = text.replace("as_of: 2026-07-12", "as_of: someday")
    path.write_text(text, encoding="utf-8")

    misplaced_dir = tmp_path / "08-Examples" / "cases"
    misplaced_dir.mkdir(parents=True, exist_ok=True)
    (misplaced_dir / "misplaced.md").write_text(text, encoding="utf-8")

    result = validator.validate_vault(tmp_path)
    errors = "\n".join(result["errors"])
    assert "public evidence cases require attributable source_refs" in errors
    assert "case as_of must use YYYY-MM-DD" in errors
    assert "public evidence cases must exist under 09-Cases/" in errors


def test_public_evaluation_fixture_is_fictional_and_has_ten_cases():
    benchmark = (ROOT / "08-Examples" / "evaluation" / "benchmark-v1.md").read_text(encoding="utf-8")

    assert "fictional: true" in benchmark
    assert "case_count: 10" in benchmark
    assert sum(1 for line in benchmark.splitlines() if line.startswith("## E")) == 10
    assert "private/evaluations/" in benchmark


def test_model_proposal_template_requires_private_human_review():
    template = (ROOT / "03-Templates" / "model_change_proposal_template.md").read_text(encoding="utf-8")

    for field in [
        "type: model_change_proposal",
        "proposal_type:",
        "status: proposed",
        "target_model_ids:",
        "source_refs:",
        "privacy: private",
        "## Human Review",
    ]:
        assert field in template
