"""Create a candidate LollaNet mental model card."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

try:
    from model_taxonomy import canonicalize, model_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_taxonomy import canonicalize, model_path


MODEL_KINDS = {"world", "reasoning", "action"}
MODEL_ID_RE = re.compile(r"^mm-[a-z0-9]+(?:-[a-z0-9]+)*$")
PROPOSAL_ID_RE = re.compile(r"^mp-[a-z0-9]+(?:-[a-z0-9]+)*$")

DEFAULT_RELATED = {
    "激励机制": ["机会成本", "代理问题", "反馈回路"],
    "机会成本": ["能力圈", "安全边际", "期望值"],
    "安全边际": ["反向思考", "基础概率", "冗余"],
    "能力圈": ["过度自信", "基础概率", "反向思考"],
    "反向思考": ["安全边际", "确认偏误", "基础概率"],
}

DEFAULT_CONFLICTS = {
    "机会成本": ["mm-loss-aversion"],
    "安全边际": ["mm-overconfidence"],
    "能力圈": ["mm-overconfidence"],
    "反向思考": ["mm-confirmation-bias"],
}


def default_model_id(
    name: str,
    discipline: str,
    source_category: str = "学科知识",
    subdiscipline: str = "通用",
) -> str:
    """Return a deterministic initial ID; the persisted ID must survive later renames."""
    classification = canonicalize(source_category, discipline, subdiscipline)
    source = (
        f"{classification.source_category}\0{classification.discipline}\0"
        f"{classification.subdiscipline}\0{name.strip()}"
    ).encode("utf-8")
    return f"mm-{hashlib.sha256(source).hexdigest()[:12]}"


def yaml_list(items: list[str]) -> str:
    return " []" if not items else "\n" + "\n".join(f"  - {item}" for item in items)


def wiki_list(items: list[str]) -> str:
    return "- 暂无" if not items else "\n".join(f"- [[{item}]]" for item in items)


def frontmatter_value(text: str, key: str) -> str | None:
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return None
    block = text[4:].split("\n---\n", 1)[0]
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*?)[ \t]*$", block)
    return None if not match else match.group(1).strip().strip('"\'')


def frontmatter_list(text: str, key: str) -> list[str]:
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return []
    block = text[4:].split("\n---\n", 1)[0]
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*?)\s*$", block)
    if not match:
        return []
    inline = match.group(1).strip()
    if inline.startswith("[") and inline.endswith("]"):
        return [item.strip().strip('"\'') for item in inline[1:-1].split(",") if item.strip()]
    if inline:
        return [inline.strip('"\'')]
    values: list[str] = []
    for line in block[match.end() :].splitlines():
        item = re.fullmatch(r"\s{2,}-\s+(.+?)\s*", line)
        if item:
            values.append(item.group(1).strip('"\''))
        elif line.strip():
            break
    return values


def accepted_new_proposal(vault_root: Path, proposal_id: str) -> Path:
    if not PROPOSAL_ID_RE.fullmatch(proposal_id):
        raise ValueError("proposal_id must be an ASCII identifier beginning with 'mp-'")

    accepted_root = Path(vault_root) / "private" / "model-proposals" / "accepted"
    matches: list[Path] = []
    if accepted_root.exists():
        for path in accepted_root.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            if frontmatter_value(text, "id") == proposal_id:
                matches.append(path)

    if not matches:
        raise ValueError(f"accepted new-model proposal not found: {proposal_id}")
    if len(matches) > 1:
        raise ValueError(f"duplicate accepted proposal id: {proposal_id}")

    path = matches[0]
    text = path.read_text(encoding="utf-8")
    if frontmatter_value(text, "type") != "model_change_proposal":
        raise ValueError(f"accepted proposal has wrong type: {proposal_id}")
    if frontmatter_value(text, "proposal_type") != "new":
        raise ValueError(f"proposal {proposal_id} is not a new-model proposal")
    if frontmatter_value(text, "status") != "accepted":
        raise ValueError(f"proposal {proposal_id} is not accepted")
    if frontmatter_value(text, "privacy") != "private":
        raise ValueError(f"proposal {proposal_id} must remain private")
    return path


def model_body(
    name: str,
    discipline: str,
    importance: int,
    frequency: str,
    today: str,
    english_name: str | None = None,
    model_id: str | None = None,
    kind: str = "reasoning",
    proposal_id: str = "accepted-proposal-required",
    source_category: str = "学科知识",
    subdiscipline: str = "通用",
    formalism: bool = False,
) -> str:
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(MODEL_KINDS))}")

    classification = canonicalize(source_category, discipline, subdiscipline)
    model_id = model_id or default_model_id(
        name,
        classification.discipline,
        classification.source_category,
        classification.subdiscipline,
    )
    if not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id must be an ASCII identifier beginning with 'mm-'")

    related = DEFAULT_RELATED.get(name, ["激励机制", "机会成本", "反向思考"])
    conflicts = DEFAULT_CONFLICTS.get(name, [])
    confusing = related[:2]
    correctors = related[1:3] if len(related) >= 3 else related
    english_name = english_name or "Needs English Name"
    return f"""---
type: mental_model
id: {model_id}
name: {name}
english_name: {english_name}
discipline: {classification.discipline}
source_category: {classification.source_category}
subdiscipline: {classification.subdiscipline}
formalism: {str(formalism).lower()}
kind: {kind}
status: candidate
version: 1
importance: {importance}
frequency: {frequency}
mastery: 1
created: {today}
updated: {today}
tags:
  - mental-model
  - decision-making
related:{yaml_list(related)}
conflicts:{yaml_list(conflicts)}
supersedes: []
use_cases:
  - 重大决策
  - 商业判断
  - 个人复盘
---

# {name}

## 一句话定义

候选卡尚需补充：这个模型检查什么变量，以及它如何改变判断。

## 核心机制

候选卡尚需补充可检验的因果链、关键变量，以及增强或削弱机制的条件。

## 形式化内容

{"候选卡尚需补充定义、符号、公式或算法、成立前提、推导或证明来源、反例与跨域转译。" if formalism else "本卡不要求独立的形式化表示；如后续发现公式、算法或可证明结构会改变判断，应提出模型修订提案。"}

## 耐久机制

识别不依赖特定案例、行业或时代的底层约束；接受本卡前需要说明该机制为何能跨情境保持解释力。

## 时代性表现

制度、技术、成本结构和信息环境可能改变模型的表现形式与强度。历史类比必须逐项核验这些差异。

## 芒格式解释

来源需要核验。这里只能补充可操作转述，不得伪造芒格原话。

## 适用边界

适用：
- 问题需要把抽象概念转为可验证变量。

不适用：
- 事实尚未厘清，只是在套概念。
- 关键变量无法观察，也没有可用代理指标。

容易误判的边界：
- 模型只能提出检查问题，不能替代证据。
- 模型与基础事实冲突时，先回到事实。

## AI 调用条件

- 问题包含方向、项目、投资、合作或组织取舍。
- 模型可以改变至少一个假设、预测或行动。
- 当前判断存在概念混用、证据不足或时间尺度遗漏。

## 不应调用

- 用户只是在询问事实，尚未进入判断或取舍。
- 缺少基本背景，连问题类型都还没有定义。
- 模型只会重复已有结论，不能增加新检查项。

## 场景化决策提问

### 职业方向

1. 这个模型检查职业路线中的哪个关键变量？
2. 哪些证据支持或反对当前判断？
3. 三个月内能用什么结果验证它？

### 创业/产品

1. 这个模型对应哪个用户行为或业务指标？
2. 如果它成立，最小实验应该验证什么？
3. 如果它不成立，产品最先在哪里失效？

### 投资/商业

1. 这个模型改变的是概率、收益、风险还是时间尺度？
2. 同类案例的基础概率和边界是什么？
3. 什么证据会触发退出或更新判断？

### 合作/组织

1. 各方是否共享模型所检查的变量？
2. 谁掌握信息，谁获得收益，谁承担代价？
3. 模型会不会掩盖执行、能力或激励问题？

## 典型误用

- 把模型当成口号或答案。
- 只寻找支持已有判断的证据。
- 忽略基础事实、时代条件和适用边界。

## 反向思考

如果这个模型被误用，最可能把判断带向哪个错误结论？

## 经典案例

### 案例事实

暂无可靠案例；候选卡接受前需要补充可核验材料。

### 模型解释

需要说明案例如何体现机制，并分离机制、制度、技术和时代条件。

### 来源

- 核心来源：待验证线索；接受前替换为 `[作者或机构，标题（年份）](可回引 URL 或 DOI)`。
- 案例来源：待验证线索；接受前补充观察日期和可回引 URL。
- 来源支持的事实：暂无。
- 来源没有证明的推断：当前案例与机制解释均未验证。

## 支持证据

- 机制主张：暂无足以激活本模型的可靠证据。
  - 来源：待验证线索。
  - 证据类型：未分类。
  - 支持范围：暂无。
  - 外推限制：暂无证据支持跨场景外推。
- 独立支持或复现：暂无；接受前至少补充一项独立材料。

## 反证

- 最强反证或边界来源：待验证线索。
- 削弱模型的观察：关键变量不存在时，本模型可能只是事后解释。
- 可替代解释：其他机制同样能解释结果时，必须降低本模型置信度。
- 转为 `contested` 的条件：可靠反证显示预测方向不稳定或依赖未声明的边界。

## 失效条件

- 任务中缺少该模型声称的必要机制。
- 当前指标无法区分本模型与主要替代解释。
- 制度、技术、样本或时间变化破坏了来源材料的外推条件。

## 可观察预测

- 预测对象与时间窗：接受前定义。
- 如果模型成立：关键变量的变化应先于或伴随目标结果出现。
- 区分性结果：接受前写明与主要替代解释不同的预测。
- 降低置信度的结果：重复观察未出现预测关系，或替代解释预测得更准确。

## 关联模型

### 常一起调用

{wiki_list(related)}

### 容易混淆

{wiki_list(confusing)}

### 互相校正

{wiki_list(correctors)}

## AI 输出要求

- 关键变量：本模型实际检查什么。
- 当前证据与反证：底层来源分别支持什么。
- 边界与不确定性：什么条件会改变结论。
- 可观察预测：未来出现什么应上调或下调置信度。
- 下一步验证动作：接下来查什么、问谁或做什么实验。

## 修订历史

| 日期 | 版本 | 状态 | 变更 | 依据或提案 |
| --- | --- | --- | --- | --- |
| {today} | 1 | candidate | 创建候选模型 | {proposal_id} |
"""


def create_model(
    vault_root: Path,
    name: str,
    discipline: str = "Psychology",
    importance: int = 3,
    frequency: str = "medium",
    english_name: str | None = None,
    today: str | None = None,
    model_id: str | None = None,
    kind: str = "reasoning",
    proposal_id: str | None = None,
    source_category: str = "学科知识",
    subdiscipline: str = "通用",
    formalism: bool = False,
) -> Path:
    today = today or date.today().isoformat()
    classification = canonicalize(source_category, discipline, subdiscipline)
    model_id = model_id or default_model_id(
        name,
        classification.discipline,
        classification.source_category,
        classification.subdiscipline,
    )
    if not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id must be an ASCII identifier beginning with 'mm-'")
    target = model_path(vault_root, classification, name)
    target_dir = target.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    if proposal_id is None:
        raise ValueError("creating a public candidate requires --proposal <accepted-new-proposal-id>")
    proposal_path = accepted_new_proposal(vault_root, proposal_id)
    proposal_text = proposal_path.read_text(encoding="utf-8")
    is_batch = frontmatter_value(proposal_text, "batch") == "true"
    proposed_models = frontmatter_list(proposal_text, "proposed_models")
    if is_batch and name not in proposed_models:
        raise ValueError(f"model {name!r} is not listed in batch proposal {proposal_id}")
    for existing in (Path(vault_root) / "01-Models").rglob("*.md"):
        existing_text = existing.read_text(encoding="utf-8")
        if not is_batch and f"| {proposal_id} |" in existing_text:
            raise ValueError(f"accepted proposal already created a model: {proposal_id}")
        if frontmatter_value(existing_text, "id") == model_id:
            raise ValueError(f"model_id already exists in {existing}: {model_id}")
    target.write_text(
        model_body(
            name,
            discipline,
            importance,
            frequency,
            today,
            english_name,
            model_id,
            kind,
            proposal_id,
            classification.source_category,
            classification.subdiscipline,
            formalism,
        ),
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a candidate LollaNet mental model card.")
    parser.add_argument("name", help="Model name, for example: 激励机制")
    parser.add_argument("--source-category", default="学科知识", help="Canonical source category")
    parser.add_argument("--discipline", default="心理学", help="Canonical field; legacy English aliases are accepted")
    parser.add_argument("--subdiscipline", default="通用", help="Specific field within the discipline")
    parser.add_argument("--formalism", action="store_true", help="Require the formal-content section")
    parser.add_argument("--kind", choices=sorted(MODEL_KINDS), default="reasoning", help="Model role")
    parser.add_argument("--id", dest="model_id", default=None, help="Stable ASCII ID beginning with mm-")
    parser.add_argument(
        "--proposal",
        dest="proposal_id",
        required=True,
        help="Accepted private new-model proposal ID beginning with mp-",
    )
    parser.add_argument("--importance", type=int, default=3, help="Importance score from 1 to 5")
    parser.add_argument("--frequency", default="medium", help="Usage frequency label")
    parser.add_argument("--english-name", default=None, help="English name for the model")
    parser.add_argument("--root", default=".", help="Vault root path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = create_model(
        Path(args.root),
        args.name,
        args.discipline,
        args.importance,
        args.frequency,
        args.english_name,
        model_id=args.model_id,
        kind=args.kind,
        proposal_id=args.proposal_id,
        source_category=args.source_category,
        subdiscipline=args.subdiscipline,
        formalism=args.formalism,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
