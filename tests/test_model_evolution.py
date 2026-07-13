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
    paths = sorted((ROOT / "01-Models").rglob("*.md"))

    assert len(paths) == 30
    ids: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        block = frontmatter(text)
        model_id = scalar(block, "id")

        assert validator.MODEL_ID_RE.fullmatch(model_id)
        assert model_id not in ids
        ids.add(model_id)
        assert scalar(block, "kind") in validator.MODEL_KINDS
        assert scalar(block, "status") in validator.MODEL_STATUSES
        assert int(scalar(block, "version")) >= 1
        assert "conflicts:" in block
        assert "supersedes:" in block
        for section in validator.MODEL_SECTIONS:
            assert section in text, (path.name, section)
        assert "## 使用记录" not in text
        assert "当前个人使用线索" not in text


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
        module.model_body("坏模型", "Test", 3, "low", "2026-07-12", kind="unknown")
    with pytest.raises(ValueError, match="model_id"):
        module.model_body("坏模型", "Test", 3, "low", "2026-07-12", model_id="not stable")


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
        discipline="Test",
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
