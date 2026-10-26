import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "06-Scripts" / "build_game_design_cards.py"
DATA = ROOT / "05-Maps" / "Game Design" / "adapters.yaml"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_game_design_cards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_public_model_has_one_game_design_adapter():
    builder = load_builder()
    models = builder.load_models(ROOT)
    adapters = builder.load_adapters(DATA)

    builder.validate_adapters(models, adapters)

    assert len(models) == 88
    assert len(adapters) == 88
    assert {item["source_model_id"] for item in adapters} == set(models)


def test_adapters_are_specific_and_mark_transfer_strength():
    builder = load_builder()
    models = builder.load_models(ROOT)
    adapters = builder.load_adapters(DATA)

    assert Counter(item["transfer_mode"] for item in adapters) == {
        "direct": 66,
        "structural": 8,
        "mechanic-only": 14,
    }
    assert {models[item["source_model_id"]]["discipline"] for item in adapters} == {
        "制度演化",
        "概率与统计",
        "工程学",
        "化学",
        "经济学",
        "商学",
        "生物学",
        "数学",
        "物理学",
        "心理学",
        "哲学",
    }
    for field in ["translation", "decision", "prototype", "falsifier"]:
        assert len({item[field] for item in adapters}) == 88


def test_generated_cards_are_current_and_traceable():
    builder = load_builder()
    models = builder.load_models(ROOT)
    adapters = builder.load_adapters(DATA)
    outputs = builder.expected_outputs(ROOT, models, adapters)

    assert len(outputs) == 89
    for path, expected in outputs.items():
        assert path.read_text(encoding="utf-8") == expected

    for adapter in adapters:
        model = models[adapter["source_model_id"]]
        path = (
            ROOT
            / "05-Maps"
            / "Game Design"
            / "Cards"
            / model["discipline"]
            / f"{model['name']}.md"
        )
        text = path.read_text(encoding="utf-8")
        metadata = builder.parse_frontmatter(text, path)
        assert metadata["source_model_id"] == model["id"]
        assert metadata["source_model_version"] == model["version"]
        assert metadata["transfer_mode"] == adapter["transfer_mode"]
        assert "\n...\n" not in text
        for section in [
            "## 母模型",
            "## 设计机制",
            "## 适用域",
            "## 关键决策",
            "## 最小原型",
            "## 观察与反证",
            "## 误用边界",
        ]:
            assert section in text
        assert f"# {model['name']}\n" in text
        assert f"# {model['name']}（游戏策划）" not in text

    index = outputs[ROOT / "05-Maps" / "Game Design" / "README.md"]
    assert "# LollaNet 游戏策划模型" in index
    assert "游戏策划适配卡" not in index
