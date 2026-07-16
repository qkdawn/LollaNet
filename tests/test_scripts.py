import hashlib
import importlib.util
import json
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

    assert first == tmp_path / "01-Models" / "学科知识" / "心理学" / "激励机制.md"
    text = first.read_text(encoding="utf-8")
    assert "type: mental_model" in text
    assert "id: mm-" in text
    assert "name: 激励机制" in text
    assert "english_name: Needs English Name" in text
    assert "source_category: 学科知识" in text
    assert "discipline: 心理学" in text
    assert "subdiscipline: 通用" in text
    assert "formalism: false" in text
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
    dashboard = tmp_path / "private" / "me" / "dashboard.md"
    value_function = tmp_path / "private" / "me" / "Value Function.md"
    goals = tmp_path / "private" / "me" / "Long-term Goals.md"
    principles = tmp_path / "private" / "me" / "Principles.md"
    current_focus = tmp_path / "private" / "me" / "Current Focus.md"
    inbox = tmp_path / "private" / "projects" / "inbox"

    assert dashboard in created
    assert profile in created
    assert value_function in created
    assert goals in created
    assert principles in created
    assert current_focus in created
    assert (tmp_path / "private" / "me" / "observations").is_dir()
    assert (tmp_path / "private" / "me" / "reviews").is_dir()
    assert (tmp_path / "private" / "me" / "proposals" / "proposed").is_dir()
    assert (inbox / "Project Profile.md").exists()
    assert (inbox / "decisions" / "open").is_dir()

    profile.write_text("custom profile", encoding="utf-8")
    value_function.write_text("custom value function", encoding="utf-8")
    module.init_private(tmp_path)

    assert profile.read_text(encoding="utf-8") == "custom profile"
    assert value_function.read_text(encoding="utf-8") == "custom value function"


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


def test_validate_vault_allows_root_entry_docs_without_frontmatter(tmp_path):
    module = load_script("validate_vault.py")
    for directory in module.REQUIRED_DIRS:
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")
    (tmp_path / "PROJECT_SPEC.md").write_text("# Project Spec\n", encoding="utf-8")

    result = module.validate_vault(tmp_path)

    assert result["error_count"] == 0


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


def test_blind_review_integrity_verifier_accepts_matching_fixture_and_rejects_tampering(tmp_path):
    module = load_script("verify_blind_review_integrity.py")
    packet = tmp_path / "packet.md"
    mapping = tmp_path / "mapping.json"
    packet.write_text("blind packet", encoding="utf-8")
    mapping.write_text('{"a": "b"}', encoding="utf-8")
    manifest = {
        "type": "independent_review_integrity_manifest",
        "waves": [
            {
                "wave": "fixture-wave",
                "blind_packet_path": packet.name,
                "blind_packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "packet_bytes": packet.stat().st_size,
                "mapping_withheld_path": mapping.name,
                "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
                "mapping_bytes": mapping.stat().st_size,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    clean = module.verify_manifest(manifest_path)
    assert clean["passed"]
    assert clean["waves_checked"] == 1
    assert clean["artifacts_checked"] == 2

    packet.write_text("tampered packet", encoding="utf-8")
    tampered = module.verify_manifest(manifest_path)
    assert not tampered["passed"]
    assert any("SHA-256 mismatch" in error for error in tampered["errors"])


def test_source_review_integrity_verifier_accepts_matching_fixture_and_rejects_tampering(tmp_path):
    module = load_script("verify_source_review_integrity.py")
    packet = tmp_path / "source-packet.md"
    template = tmp_path / "record-template.md"
    packet.write_text("source review packet", encoding="utf-8")
    template.write_text("source review template", encoding="utf-8")
    manifest = {
        "type": "independent_source_review_integrity_manifest",
        "waves": [
            {
                "wave": "fixture-wave",
                "source_packet_path": packet.name,
                "source_packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "packet_bytes": packet.stat().st_size,
                "review_record_template_path": template.name,
                "review_record_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                "template_bytes": template.stat().st_size,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    clean = module.verify_manifest(manifest_path)
    assert clean["passed"]
    assert clean["waves_checked"] == 1
    assert clean["artifacts_checked"] == 2

    template.write_text("tampered source review template", encoding="utf-8")
    tampered = module.verify_manifest(manifest_path)
    assert not tampered["passed"]
    assert any("SHA-256 mismatch" in error for error in tampered["errors"])


def test_retrospective_task_integrity_verifier_accepts_fixture_and_rejects_tampering(tmp_path):
    module = load_script("verify_retrospective_task_integrity.py")
    charter = tmp_path / "charter.md"
    packet = tmp_path / "packet.md"
    rubric = tmp_path / "rubric.md"
    charter.write_text("frozen charter", encoding="utf-8")
    packet.write_text("frozen packet", encoding="utf-8")
    rubric.write_text("frozen rubric", encoding="utf-8")
    artifacts = []
    for artifact in [charter, packet, rubric]:
        artifacts.append(
            {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "bytes": artifact.stat().st_size,
            }
        )
    manifest = {
        "type": "source_grounded_retrospective_integrity_manifest",
        "artifacts": artifacts,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    clean = module.verify_manifest(manifest_path)
    assert clean["passed"]
    assert clean["artifacts_checked"] == 3

    rubric.write_text("tampered rubric", encoding="utf-8")
    tampered = module.verify_manifest(manifest_path)
    assert not tampered["passed"]
    assert any("SHA-256 mismatch" in error for error in tampered["errors"])


def test_source_review_coverage_checker_accepts_fixture_and_rejects_coverage_or_hash_tampering(tmp_path):
    module = load_script("check_source_review_coverage.py")
    packet = tmp_path / "wave" / "source-packet.md"
    template = tmp_path / "wave" / "review-template.md"
    packet.parent.mkdir(parents=True)
    packet.write_text("### Fixture model\n", encoding="utf-8")
    template.write_text("### Fixture review record\n", encoding="utf-8")

    (tmp_path / "12-all-model-coverage.csv").write_text(
        "model_id,wave,quality_gate\n"
        "mm-fixture,fixture-wave,"
        "same_reviewer_direct_source_locator_captured; independent_content_review_pending\n",
        encoding="utf-8",
    )
    (tmp_path / "27-all-model-core-claim-source-locator-gap.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_id": "mm-fixture",
                        "source_claim_status": "exact_locator_captured_pending_independent_content_review",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "54-independent-source-content-review-handoff-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {"wave": "fixture-wave", "models": 1, "model_ids": ["mm-fixture"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    integrity = {
        "waves": [
            {
                "wave": "fixture-wave",
                "models": 1,
                "source_packet_path": "wave/source-packet.md",
                "source_packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "packet_bytes": packet.stat().st_size,
                "review_record_template_path": "wave/review-template.md",
                "review_record_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                "template_bytes": template.stat().st_size,
            }
        ]
    }
    integrity_path = tmp_path / "55-independent-source-content-review-integrity-manifest.json"
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")

    clean = module.check_campaign(tmp_path)
    assert clean["passed"]
    assert clean["models_checked"] == 1
    assert clean["waves_checked"] == 1

    (tmp_path / "12-all-model-coverage.csv").write_text(
        "model_id,wave,quality_gate\n"
        "mm-fixture,wrong-wave,"
        "same_reviewer_direct_source_locator_captured; independent_content_review_pending\n",
        encoding="utf-8",
    )
    coverage_tampered = module.check_campaign(tmp_path)
    assert not coverage_tampered["passed"]
    assert any("model wave mismatch" in error for error in coverage_tampered["errors"])

    (tmp_path / "12-all-model-coverage.csv").write_text(
        "model_id,wave,quality_gate\n"
        "mm-fixture,fixture-wave,"
        "same_reviewer_direct_source_locator_captured; independent_content_review_pending\n",
        encoding="utf-8",
    )
    template.write_text("### Tampered fixture review record\n", encoding="utf-8")
    hash_tampered = module.check_campaign(tmp_path)
    assert not hash_tampered["passed"]
    assert any("SHA-256 mismatch" in error for error in hash_tampered["errors"])


def test_source_review_submission_validator_requires_complete_distinct_frozen_records(tmp_path):
    module = load_script("validate_source_review_submissions.py")
    base = tmp_path / "campaign"
    submissions = base / "source-review-submissions" / "wave-fixture"
    submissions.mkdir(parents=True)
    (base / "54-independent-source-content-review-handoff-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {"wave": "wave-fixture", "models": 1, "model_ids": ["mm-fixture"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    packet_hash = "a" * 64
    (base / "55-independent-source-content-review-integrity-manifest.json").write_text(
        json.dumps(
            {"waves": [{"wave": "wave-fixture", "source_packet_sha256": packet_hash}]}
        ),
        encoding="utf-8",
    )

    def submission(reviewer_id: str, submission_id: str):
        return {
            "type": "independent_source_content_review_submission",
            "campaign": "model-test-v2-2026-07-16",
            "status": "frozen_submitted",
            "submission_id": submission_id,
            "wave": "wave-fixture",
            "reviewer": {
                "reviewer_id": reviewer_id,
                "review_date": "2026-07-16",
                "agent_or_tool": "independent human reviewer",
                "independence_declaration": {
                    "participated_in_lollanet_card_authoring": False,
                    "participated_in_source_location_or_prior_audit": False,
                    "participated_in_task_or_answer_generation": False,
                    "participated_in_governance_for_this_wave": False,
                    "conflict_statement": "none declared",
                },
            },
            "source_packet_sha256": packet_hash,
            "records": [
                {
                    "model_id": "mm-fixture",
                    "disposition": "supports_narrow_claim",
                    "source_refs": [
                        {
                            "stable_url": "https://example.test/source",
                            "version_or_access_date": "2026-07-16",
                            "locator": "p. 1",
                        }
                    ],
                    "support_or_limit": "Narrow claim supported.",
                    "counterevidence_or_boundary": "Applies only under the stated assumptions.",
                    "access_notes": "Public source accessed.",
                    "confidence": "medium",
                }
            ],
            "wave_summary": {
                "recommended_next_step": "continue_research",
                "unresolved_disagreements": "none known",
            },
        }

    first_path = submissions / "r01-2026-07-16.json"
    first_path.write_text(json.dumps(submission("r01", "submission-r01")), encoding="utf-8")
    ready = module.validate_campaign(base)
    assert ready["passed"]
    assert ready["valid_submission_count"] == 1
    assert not ready["completion_format_gate_met"]

    insufficient = module.validate_campaign(base, require_two_per_wave=True)
    assert not insufficient["passed"]
    assert any("fewer than two" in error for error in insufficient["errors"])

    second_path = submissions / "r02-2026-07-16.json"
    second_path.write_text(json.dumps(submission("r02", "submission-r02")), encoding="utf-8")
    complete = module.validate_campaign(base, require_two_per_wave=True)
    assert complete["passed"]
    assert complete["completion_format_gate_met"]

    tampered = submission("r03", "submission-r03")
    tampered["source_packet_sha256"] = "b" * 64
    (submissions / "r03-2026-07-16.json").write_text(json.dumps(tampered), encoding="utf-8")
    invalid_hash = module.validate_campaign(base)
    assert not invalid_hash["passed"]
    assert any("source_packet_sha256" in error for error in invalid_hash["errors"])

    (submissions / "r03-2026-07-16.json").unlink()
    duplicate = submission("r01", "submission-r01-copy")
    (submissions / "r01-copy-2026-07-16.json").write_text(json.dumps(duplicate), encoding="utf-8")
    duplicate_reviewer = module.validate_campaign(base)
    assert not duplicate_reviewer["passed"]
    assert any("duplicate reviewer_id" in error for error in duplicate_reviewer["errors"])



def test_blind_review_submission_validator_requires_complete_distinct_frozen_records(tmp_path):
    module = load_script("validate_blind_review_submissions.py")
    base = tmp_path / "campaign"
    submissions = base / "blind-review-submissions" / "wave-fixture"
    submissions.mkdir(parents=True)
    (base / "16-independent-review-handoff-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "wave": "wave-fixture",
                        "models": 1,
                        "unit_ids": ["U01", "U02"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    packet_hash = "b" * 64
    (base / "28-independent-review-integrity-manifest.json").write_text(
        json.dumps({"waves": [{"wave": "wave-fixture", "blind_packet_sha256": packet_hash}]}),
        encoding="utf-8",
    )

    def submission(reviewer_id: str, submission_id: str):
        return {
            "type": "independent_blind_review_submission",
            "campaign": "model-test-v2-2026-07-16",
            "status": "frozen_submitted",
            "submission_id": submission_id,
            "wave": "wave-fixture",
            "reviewer": {
                "reviewer_id": reviewer_id,
                "review_date": "2026-07-16",
                "agent_or_tool": "fixture-reviewer",
                "independence_declaration": {
                    "participated_in_lollanet_card_authoring": False,
                    "participated_in_task_or_answer_generation": False,
                    "participated_in_prior_blind_review_or_scoring": False,
                    "saw_randomization_mapping_before_scoring": False,
                    "saw_full_ablation_mapping_before_scoring": False,
                    "used_private_user_data": False,
                    "conflict_statement": "none declared",
                },
            },
            "blind_packet_sha256": packet_hash,
            "records": [
                {
                    "unit_id": "U01",
                    "preference": "A",
                    "correctness": "A",
                    "boundary_handling": "A",
                    "action_increment": "A",
                    "unsupported_claims": "neither",
                    "rationale": "A provides a bounded action increment.",
                },
                {
                    "unit_id": "U02",
                    "preference": "tie",
                    "correctness": "tie",
                    "boundary_handling": "tie",
                    "action_increment": "tie",
                    "unsupported_claims": "neither",
                    "rationale": "No material task-level distinction.",
                },
            ],
            "wave_summary": {
                "unresolved_disagreements": "none known",
                "limitations": "fixture only",
            },
        }

    first = submission("r01", "blind-fixture-r01")
    first_path = submissions / "r01-2026-07-16.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")

    one_valid = module.validate_campaign(base)
    assert one_valid["passed"]
    assert one_valid["format_valid_submissions"] == 1
    assert one_valid["declared_independent_reviewer_counts"] == {"wave-fixture": 1}

    missing_second = module.validate_campaign(base, require_two_per_wave=True)
    assert not missing_second["passed"]
    assert any("fewer than two format-valid" in error for error in missing_second["errors"])

    second_path = submissions / "r02-2026-07-16.json"
    second_path.write_text(json.dumps(submission("r02", "blind-fixture-r02")), encoding="utf-8")
    two_valid = module.validate_campaign(base, require_two_per_wave=True)
    assert two_valid["passed"]
    assert two_valid["format_valid_submissions"] == 2
    assert two_valid["two_submission_format_gate_waves"] == ["wave-fixture"]

    changed_hash = json.loads(second_path.read_text(encoding="utf-8"))
    changed_hash["blind_packet_sha256"] = "c" * 64
    second_path.write_text(json.dumps(changed_hash), encoding="utf-8")
    hash_tampering = module.validate_campaign(base)
    assert not hash_tampering["passed"]
    assert any("blind_packet_sha256 does not match" in error for error in hash_tampering["errors"])

    second_path.write_text(json.dumps(submission("r02", "blind-fixture-r02")), encoding="utf-8")
    incomplete = json.loads(first_path.read_text(encoding="utf-8"))
    incomplete["records"] = incomplete["records"][:1]
    first_path.write_text(json.dumps(incomplete), encoding="utf-8")
    missing_unit = module.validate_campaign(base)
    assert not missing_unit["passed"]
    assert any("cover every unit_id exactly once" in error for error in missing_unit["errors"])

    first_path.write_text(json.dumps(submission("r02", "blind-fixture-r03")), encoding="utf-8")
    duplicate_reviewer = module.validate_campaign(base)
    assert not duplicate_reviewer["passed"]
    assert any("duplicate reviewer_id" in error for error in duplicate_reviewer["errors"])

def test_formalism_regression_checks_cover_all_formal_model_cards():
    module = load_script("check_model_formalisms.py")

    report = module.run_checks(ROOT)

    assert report["formal_card_count"] == 59
    assert report["defined_check_count"] == 59
    assert report["passed"]
    assert report["errors"] == []
    assert len(report["results"]) == 59
    assert all(result["passed"] for result in report["results"])





def test_blind_mapping_consistency_checker_rejects_missing_same_and_one_sided_maps(tmp_path):
    module = load_script("check_blind_mapping_consistency.py")
    mapping_path = tmp_path / "map.json"
    (tmp_path / "16-independent-review-handoff-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "wave": "wave-fixture",
                        "unit_ids": ["U01", "U02"],
                        "mapping_withheld_path": "map.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "28-independent-review-integrity-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "wave": "wave-fixture",
                        "mapping_withheld_path": "map.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    valid_map = {
        "U01": {"A": "full", "B": "ablation"},
        "U02": {"A": "ablation", "B": "full"},
    }
    mapping_path.write_text(json.dumps(valid_map), encoding="utf-8")

    clean = module.check_campaign(tmp_path)
    assert clean["passed"]
    assert clean["waves_checked"] == 1
    assert clean["units_checked"] == 2

    mapping_path.write_text(json.dumps({"U01": valid_map["U01"]}), encoding="utf-8")
    missing = module.check_campaign(tmp_path)
    assert not missing["passed"]
    assert any("mapping keys do not exactly match" in error for error in missing["errors"])

    same = {
        "U01": {"A": "full", "B": "full"},
        "U02": {"A": "ablation", "B": "full"},
    }
    mapping_path.write_text(json.dumps(same), encoding="utf-8")
    same_condition = module.check_campaign(tmp_path)
    assert not same_condition["passed"]
    assert any("must map opposite" in error for error in same_condition["errors"])

    one_sided = {
        "U01": {"A": "full", "B": "ablation"},
        "U02": {"A": "full", "B": "ablation"},
    }
    mapping_path.write_text(json.dumps(one_sided), encoding="utf-8")
    one_sided_report = module.check_campaign(tmp_path)
    assert not one_sided_report["passed"]
    assert any("orientation is one-sided" in error for error in one_sided_report["errors"])

def test_blind_packet_structure_checker_rejects_coverage_and_identity_leaks(tmp_path):
    module = load_script("check_blind_packet_structure.py")
    packet = tmp_path / "packet.md"
    packet.write_text(
        "## U01 — fixture\n\n**A：** bounded answer\n\n**B：** alternative answer\n",
        encoding="utf-8",
    )
    (tmp_path / "16-independent-review-handoff-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "wave": "wave-fixture",
                        "blind_packet": "packet.md",
                        "unit_ids": ["U01"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "28-independent-review-integrity-manifest.json").write_text(
        json.dumps(
            {
                "waves": [
                    {
                        "wave": "wave-fixture",
                        "blind_packet_path": "packet.md",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    clean = module.check_campaign(tmp_path)
    assert clean["passed"]
    assert clean["waves_checked"] == 1
    assert clean["units_checked"] == 1

    packet.write_text(
        "## U01 — fixture\n\n**A：** bounded answer\n\n**B：** alternative answer\n"
        "\n## U01 — duplicate\n\n**A：** duplicate A\n\n**B：** duplicate B\n",
        encoding="utf-8",
    )
    duplicate = module.check_campaign(tmp_path)
    assert not duplicate["passed"]
    assert any("duplicate packet unit_ids" in error for error in duplicate["errors"])

    packet.write_text(
        "## U01 — fixture\n\n**A：** full\n\n**B：** ablation\n",
        encoding="utf-8",
    )
    assignment_leak = module.check_campaign(tmp_path)
    assert not assignment_leak["passed"]
    assert any("direct A/B-to-condition assignment leak" in error for error in assignment_leak["errors"])

    packet.write_text(
        "## U01 — fixture\n\n**A：** see mm-fixture\n\n**B：** alternative answer\n",
        encoding="utf-8",
    )
    identity_leak = module.check_campaign(tmp_path)
    assert not identity_leak["passed"]
    assert any("model ID or card-path leak" in error for error in identity_leak["errors"])

def test_blind_review_delivery_preparer_exports_only_verified_mapping_free_bundle(tmp_path):
    module = load_script("prepare_blind_review_delivery.py")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    packet = source_dir / "blind-packet.md"
    mapping = source_dir / "randomization-map.json"
    packet.write_text("# Frozen blind packet\nA versus B only\n", encoding="utf-8")
    mapping.write_text('{"secret": "not for reviewers"}', encoding="utf-8")
    manifest = {
        "type": "independent_review_integrity_manifest",
        "campaign": "fixture-campaign",
        "waves": [
            {
                "wave": "wave-fixture",
                "blind_packet_path": "source/blind-packet.md",
                "blind_packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "packet_bytes": packet.stat().st_size,
                "mapping_withheld_path": "source/randomization-map.json",
            }
        ],
    }
    manifest_path = tmp_path / "integrity.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "delivery"
    prepared = module.prepare_bundle(manifest_path, "wave-fixture", output)
    assert prepared["passed"]
    assert prepared["artifacts_checked"] == 2
    assert {item.name for item in output.iterdir()} == {
        "blind-packet.md",
        "README.md",
        "delivery-manifest.json",
    }
    assert "do not seek" in (output / "README.md").read_text(encoding="utf-8").lower()
    assert "not for reviewers" not in (output / "blind-packet.md").read_text(encoding="utf-8")
    assert module.verify_bundle(output)["passed"]

    (output / "blind-packet.md").write_text("tampered", encoding="utf-8")
    tampered = module.verify_bundle(output)
    assert not tampered["passed"]
    assert any("SHA-256 mismatch" in error for error in tampered["errors"])

    output_with_map = tmp_path / "delivery-with-map"
    assert module.prepare_bundle(manifest_path, "wave-fixture", output_with_map)["passed"]
    (output_with_map / "randomization-map.json").write_text("leak", encoding="utf-8")
    leaked = module.verify_bundle(output_with_map)
    assert not leaked["passed"]
    assert any("unexpected files" in error or "forbidden mapping" in error for error in leaked["errors"])

    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["waves"][0]["blind_packet_sha256"] = "d" * 64
    bad_manifest_path = tmp_path / "bad-integrity.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    bad_source = module.prepare_bundle(bad_manifest_path, "wave-fixture", tmp_path / "bad-delivery")
    assert not bad_source["passed"]
    assert any("frozen blind packet SHA-256 mismatch" in error for error in bad_source["errors"])

def test_scripts_expose_help():
    for script in [
        "create_model.py",
        "create_case.py",
        "create_decision.py",
        "create_project.py",
        "init_private.py",
        "validate_vault.py",
        "check_model_formalisms.py",
        "verify_blind_review_integrity.py",
        "verify_source_review_integrity.py",
        "check_source_review_coverage.py",
        "verify_retrospective_task_integrity.py",
        "validate_source_review_submissions.py",
        "validate_blind_review_submissions.py",
        "prepare_blind_review_delivery.py",
        "check_blind_packet_structure.py",
        "check_blind_mapping_consistency.py",
        "migrate_model_taxonomy.py",
        "generate_model_index.py",
    ]:
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

    assert len(model_paths) >= 30

    missing: list[tuple[str, str]] = []
    weak_questions: list[tuple[str, str]] = []
    for path in model_paths:
        text = path.read_text(encoding="utf-8")
        for section in required_sections:
            if section not in text:
                missing.append((path.name, section))
        if "formalism: true" in text:
            for section in [
                "## 形式化内容",
                "### 定义与符号",
                "### 公式或算法",
                "### 成立前提",
                "### 推导或证明来源",
                "### 反例与边界",
                "### 跨域转译",
            ]:
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
