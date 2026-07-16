"""Canonical model taxonomy shared by migration, creation, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SOURCE_CATEGORIES = (
    "时势信号",
    "哲学模型",
    "历史机制",
    "他人失败",
    "学科知识",
)

CATEGORY_FIELDS = {
    "哲学模型": frozenset({"哲学"}),
    "历史机制": frozenset({"制度演化"}),
    "学科知识": frozenset(
        {
            "数学",
            "物理学",
            "化学",
            "心理学",
            "经济学",
            "概率与统计",
            "商学",
            "工程学",
            "生物学",
        }
    ),
    "时势信号": frozenset(),
    "他人失败": frozenset(),
}

LEGACY_DISCIPLINE_ALIASES = {
    "Biology": "生物学",
    "Business": "商学",
    "Chemistry": "化学",
    "Economics": "经济学",
    "Engineering": "工程学",
    "History": "制度演化",
    "Math": "数学",
    "Mathematics": "数学",
    "Physics": "物理学",
    "Philosophy": "哲学",
    "Probability": "概率与统计",
    "Psychology": "心理学",
}


@dataclass(frozen=True)
class ModelClassification:
    source_category: str
    discipline: str
    subdiscipline: str


SEED_CLASSIFICATIONS: dict[str, ModelClassification] = {
    "生态位": ModelClassification("学科知识", "生物学", "生态学"),
    "代理问题": ModelClassification("学科知识", "商学", "组织与治理"),
    "规模经济": ModelClassification("学科知识", "商学", "产业组织"),
    "护城河": ModelClassification("学科知识", "商学", "竞争战略"),
    "网络效应": ModelClassification("学科知识", "商学", "平台经济"),
    "复利": ModelClassification("学科知识", "经济学", "增长与金融"),
    "机会成本": ModelClassification("学科知识", "经济学", "决策经济学"),
    "安全边际": ModelClassification("学科知识", "工程学", "安全与可靠性"),
    "反馈回路": ModelClassification("学科知识", "工程学", "系统动力学"),
    "瓶颈": ModelClassification("学科知识", "工程学", "系统工程"),
    "冗余": ModelClassification("学科知识", "工程学", "可靠性工程"),
    "路径依赖": ModelClassification("历史机制", "制度演化", "路径依赖"),
    "反向思考": ModelClassification("哲学模型", "哲学", "决策哲学"),
    "能力圈": ModelClassification("哲学模型", "哲学", "认识论与决策"),
    "贝叶斯更新": ModelClassification("学科知识", "概率与统计", "贝叶斯统计"),
    "非线性": ModelClassification("学科知识", "概率与统计", "动力系统与非线性"),
    "基础概率": ModelClassification("学科知识", "概率与统计", "概率论"),
    "均值回归": ModelClassification("学科知识", "概率与统计", "统计推断"),
    "临界点": ModelClassification("学科知识", "概率与统计", "阈值与相变"),
    "期望值": ModelClassification("学科知识", "概率与统计", "决策理论"),
    "承诺一致性": ModelClassification("学科知识", "心理学", "社会与动机"),
    "过度自信": ModelClassification("学科知识", "心理学", "认知偏误"),
    "激励导致的偏见": ModelClassification("学科知识", "心理学", "动机性推理"),
    "激励机制": ModelClassification("学科知识", "心理学", "动机与行为"),
    "嫉妒与比较": ModelClassification("学科知识", "心理学", "社会比较"),
    "可得性偏差": ModelClassification("学科知识", "心理学", "认知偏误"),
    "权威偏差": ModelClassification("学科知识", "心理学", "社会影响"),
    "确认偏误": ModelClassification("学科知识", "心理学", "认知偏误"),
    "社会认同": ModelClassification("学科知识", "心理学", "社会影响"),
    "损失厌恶": ModelClassification("学科知识", "心理学", "行为决策"),
}


def canonical_discipline(value: str) -> str:
    """Normalize legacy English directory names to canonical Chinese fields."""
    value = value.strip()
    return LEGACY_DISCIPLINE_ALIASES.get(value, value)


def canonicalize(
    source_category: str = "学科知识",
    discipline: str = "心理学",
    subdiscipline: str = "通用",
) -> ModelClassification:
    source_category = source_category.strip()
    discipline = canonical_discipline(discipline)
    subdiscipline = subdiscipline.strip() or "通用"
    if source_category not in SOURCE_CATEGORIES:
        raise ValueError(f"unknown source_category: {source_category}")
    fields = CATEGORY_FIELDS[source_category]
    if fields and discipline not in fields:
        allowed = ", ".join(sorted(fields))
        raise ValueError(f"discipline {discipline!r} is not valid under {source_category}; expected one of: {allowed}")
    if not fields and source_category in {"时势信号", "他人失败"}:
        raise ValueError(f"{source_category} is an input/evidence route, not a formal model directory")
    return ModelClassification(source_category, discipline, subdiscipline)


def classification_for_seed(name: str) -> ModelClassification:
    try:
        return SEED_CLASSIFICATIONS[name]
    except KeyError as exc:
        raise KeyError(f"missing seed taxonomy mapping for model: {name}") from exc


def model_path(vault_root: Path, classification: ModelClassification, name: str) -> Path:
    return (
        Path(vault_root)
        / "01-Models"
        / classification.source_category
        / classification.discipline
        / f"{name}.md"
    )


def classification_from_path(path: Path, vault_root: Path) -> ModelClassification | None:
    rel = path.relative_to(vault_root)
    if len(rel.parts) < 4 or rel.parts[0] != "01-Models":
        return None
    source_category, discipline = rel.parts[1:3]
    return canonicalize(source_category, discipline, "通用")
