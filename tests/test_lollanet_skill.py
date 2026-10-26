from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".agents" / "skills" / "lollanet-analyze"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skill_is_small_and_loadable():
    skill = read(SKILL_DIR / "SKILL.md")
    metadata, _ = skill[4:].split("\n---\n", 1)
    openai = read(SKILL_DIR / "agents" / "openai.yaml")

    assert "name: lollanet-analyze" in metadata
    assert "complex causal" in metadata
    assert len(skill.splitlines()) < 60
    assert "$lollanet-analyze" in openai
    assert "allow_implicit_invocation: true" in openai


def test_skill_keeps_only_the_minimal_mechanism_workflow():
    skill = read(SKILL_DIR / "SKILL.md")

    for term in ["sourced observations", "two competing programs", "timed prediction", "falsifier", "actual call graph"]:
        assert term in skill.lower()
    for removed in ["source_section", "state_change", "counterfactuals", "residuals", "model_gaps"]:
        assert removed not in skill


def test_skill_preserves_graph_and_privacy_boundaries():
    skill = read(SKILL_DIR / "SKILL.md")

    for relation in ["requires", "composes", "competes", "corrects", "specializes"]:
        assert relation in skill
    assert "Do not save a temporary Pack" in skill
    assert "Never modify formal models" in skill


def test_old_neural_runtime_proposals_are_superseded():
    paths = [
        ROOT / "07-Docs" / "papers" / "lollanet-cross-disciplinary-model-network-proposal.md",
        ROOT / "07-Docs" / "papers" / "lollanet-v1" / "02-rgse-method-spec.md",
        ROOT / "07-Docs" / "papers" / "causal-model-activation" / "01-cladder-oracle-model-activation-draft.md",
    ]
    for path in paths:
        assert "status: superseded" in read(path)
