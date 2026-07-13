from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".agents" / "skills" / "lollanet-analyze"
SKILL_PATH = SKILL_DIR / "SKILL.md"
OPENAI_PATH = SKILL_DIR / "agents" / "openai.yaml"


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
    assert "scout's conclusion is never evidence by itself" in skill
    assert "narrowest underlying heading" in skill
    assert "document-title anchor is insufficient" in skill
    assert "Reject unsupported items" in skill
    assert "select 3-7 non-redundant models" in skill
    assert "never pad the set" in skill
    assert "majority vote" in skill
    assert "metadata conflicts and incompatible predictions" in skill


def test_skill_minimizes_private_data_and_requires_write_approval():
    skill = read_skill()

    assert "read the smallest relevant sections" in skill
    assert "Do not send private content to web search" in skill
    assert "Never write real user data to public framework folders" in skill
    assert "read-only by default" in skill
    assert "An analysis request or skill invocation is not approval" in skill
    assert "after the user explicitly approves the displayed proposal" in skill
    assert "Never directly rewrite a formal model from this skill" in skill
