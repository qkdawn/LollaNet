import yaml

from research.phase1.generate_index import load_models, render
from research.phase1.validate_models import ROOT, validate


def test_phase1_model_assets_validate_and_cover_five_domains():
    assert validate() == []
    models = load_models()
    assert len(models) == 20
    assert {model["domain"] for model in models} == {
        "math",
        "physics",
        "chemistry",
        "engineering",
        "finance",
    }
    assert all(sum(model["domain"] == domain for model in models) == 4 for domain in {
        "math", "physics", "chemistry", "engineering", "finance"
    })


def test_phase1_assets_are_self_contained_and_have_composition_examples():
    forbidden = {"source_card", "public_card", "public_model_version", "numeric_parameters"}
    for path in (ROOT / "models").glob("*.yaml"):
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not forbidden & set(model)
        assert model["composition"]["composition_patterns"]
        assert all(model["id"] in pattern["route"] for pattern in model["composition"]["composition_patterns"])


def test_generated_index_matches_asset_catalog():
    generated = render(load_models())
    committed = (ROOT / "MODEL_INDEX.md").read_text(encoding="utf-8")
    assert committed == generated
    assert generated.count("| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |") == 5
