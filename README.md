---
type: project_home
title: LollaNet README
updated: 2026-07-13
---

# LollaNet

LollaNet 是一个由人拥有、可演化、可装配到不同 AI 上的长期认知层。

> 人通过 Obsidian 维护长期身份、个人准则和可演化模型；AI 在具体任务中进行可审计的多路召回、冲突分析与长期检查，并且只能提交由人审批的认知修改建议。

查理·芒格的多元思维模型是重要启发，但不是项目边界。LollaNet 也吸收阅读、历史、失败案例、跨学科研究和个人观察中的耐久机制。它关心的不是模型数量，而是一个模型是否改变了当前任务的假设、预测或行动。

## 三个待验证假设

LollaNet 不预先宣称解决了 AI 的平均化或短视问题。第一阶段验证以下假设：

1. **降低平均化**：经过人工筛选的准则、异质模型与反例召回，能让 AI 产出比常规回答更贴合问题，并提供非显然但有行动价值的增量。
2. **降低短视**：基础概率、历史机制和多个时间尺度的联合检查，能减少只优化眼前结果的建议，并在 30/90 天复盘中改善判断质量。
3. **外部认知可以改善推理**：无需重新训练模型，基于文件的 Agentic Retrieval 也能把人的长期认知资产稳定地装配到不同 AI 任务中。

这些假设必须通过对照任务、盲评和延迟复盘获得支持；在此之前只能报告机制和观察结果。

## 系统边界

```text
公开框架                    私人长期认知层                 当前任务环境
00-Start..09-Cases          private/                      ChatGPT / Codex / Claude / 其他工具
模型、协议、模板、公开案例、Skill  身份、准则、私人证据、提案、可选项目  事实、文件、聊天、执行过程、临时结果
```

- 当前真实任务默认留在它原本发生的环境中，不要求复制进 Obsidian。
- 只有用户选择长期保留的抽象，才写入 `private/`。
- AI 默认只读长期认知层，并只读取能显著影响当前判断的最小上下文。
- AI 可以质疑现有模型并展示修改建议，但未经明确批准不得改写正式认知资产。
- 公共目录不得包含用户画像、真实决策、未公开项目证据或私人分析结果；基于公开可核验来源的真实案例可以进入 `09-Cases/`。

LollaNet 不是：

- 全量个人工作区或聊天归档系统；
- 能保证正确答案的真理机器；
- 自动定义用户人格或目标的系统；
- 能静默改写自身规则和用户模型的 Agent；
- 用模型数量、投票或固定权重替代判断的组合策略。

## 长期认知资产

| 资产 | 回答的问题 | 默认位置 |
| --- | --- | --- |
| 身份 | 我是谁、具有什么经历与约束 | `private/me/User Profile.md`、`Long-term Goals.md` |
| 准则 | 我愿意怎样取舍、保护什么 | `private/me/Principles.md` |
| 模型 | 世界如何运行、如何分析、如何行动 | `01-Models/`，私人补充位于 `private/models/` |
| 证据 | 为什么相信或质疑某项认知 | 私人证据位于 `private/projects/<project_id>/evidence/`；公开来源案例位于 `09-Cases/` |
| 修改提案 | 长期认知可能怎样变化 | `private/model-proposals/<status>/` |
| 评估 | LollaNet 是否真正增加判断价值 | `private/evaluations/` |

身份是事实性描述，准则是规范性选择，模型是可证伪的解释或方法；三者不能互相冒充。一次阅读或失败只能形成候选提案，不能直接成为正式模型。

## 快速开始

1. 用 Obsidian 打开仓库根目录，并从 `00-Start/Dashboard.md` 进入。
2. 初始化本地私有层；命令只补充缺失文件，不覆盖已有内容：

```bash
python 06-Scripts/init_private.py
```

3. 在 `private/me/` 中按需维护身份、长期目标和个人准则。
4. 在 Codex 中显式调用参考 Skill：

```text
$lollanet-analyze <当前问题或任务>
```

5. 只有需要长期沉淀时，才创建项目、决策或案例：

```bash
python 06-Scripts/create_project.py <project_id> --name "<项目名称>"
python 06-Scripts/create_decision.py "<决策标题>" --project <project_id>
python 06-Scripts/create_case.py "<私人案例标题>" --project <project_id>
python 06-Scripts/create_case.py "<公开案例标题>" --public-case --source "<可核验来源>"
python 06-Scripts/create_case.py "<虚构示例标题>" --public-example
```

普通事实查询、格式转换、低风险执行任务不必启动多路召回，也不必写回 LollaNet。

## 分析方式

通用流程以 `04-Prompts/LollaNet Analysis Protocol.md` 为唯一事实源，且不绑定 AI 厂商：

1. 建模当前任务；
2. 选择最小必要的私人上下文；
3. 从模型、记忆与挑战三个方向召回；
4. 去重并分析模型冲突；
5. 检查基础概率、证据质量和不确定性；
6. 比较多个时间尺度；历史类比还必须比较机制与时代条件；
7. 用适合当前任务的自然结构回答；
8. 可选展示长期修改提案。

一次综合只保留 3–7 个非重复且真正改变判断的模型。Scout 的总结不是证据，必须回引底层文件、章节或外部来源。

## 知识来源分类体系与外部认知输入（未来能力）

LollaNet 将促进模型迭代的知识来源组织为三级分类体系：

```text
知识来源分类体系
├── 一级分类：知识来源大类
│   ├── 时势信号：实时变化、外部信息流和正在发生的事件
│   ├── 哲学模型：哲学思想、概念框架和关于判断与行动的根本问题
│   ├── 历史机制：历史事件背后的制度、激励和重复性机制
│   ├── 他人失败：他人的失败案例、反例、警示和代价记录
│   └── 学科知识：各学科研究、专业知识和可验证的解释
├── 二级分类：一级分类下的具体领域
│   └── 例如：学科知识 / 心理学、经济学、生物学
└── 三级分类：具体模型
    └── 例如：学科知识 / 心理学 / 确认偏误
```

一级分类和二级分类描述知识来源，三级分类才是正式模型。原始 RSS 条目、文章、研究和失败案例先作为输入材料或证据保存，不应因为进入系统就自动成为模型。分析确认其改变了假设、预测或行动后，才将其关联到已有模型，或生成模型修改提案。同一份材料可以有一个主要来源路径，并关联多个模型。

LollaNet 计划先为“时势信号”增加一条可审计的外部输入闭环：从用户选择的信息源获取 RSS，借助 Horizon 等收集层和 `qiaomu-anything-to-notebooklm` 等内容获取/转换工具，把来源明确的材料送入暂存区，再由 LollaNet 生成“模型迭代意见日报”。日报只报告值得人工检查的机制、反证、候选模型修改和待验证问题，不自动修改模型，也不把每篇文章都写入长期层。

该方向的边界、数据契约和验收标准见 [[PROJECT_SPEC]] 的“外部认知输入闭环”，实施草案见 [`07-Docs/plans/external-input-loop.md`](07-Docs/plans/external-input-loop.md)。具体订阅源、账号、偏好和日报属于私人项目资料，不进入公共框架。

## 模型演化

正式模型使用四种状态：`candidate`、`active`、`contested`、`deprecated`。废弃只表示不再默认使用，历史记录仍保留。

AI 可提出六种修改：`new`、`amend`、`split`、`merge`、`contest`、`deprecate`。提案状态为 `proposed`、`accepted`、`rejected` 或 `deferred`。只有用户接受提案后，才能创建候选模型或递增已有模型版本。

案例按材料属性分流，三个创建模式互斥：

- `--public-case --source "<ref>"`：基于公开可核验来源的真实案例，写入 `09-Cases/<domain>/`。`source_refs` 必须非空，并分别记录事件日期 `date_observed` 与资料核验截止日 `as_of`；正文必须区分事实与推断及时代条件。引用并定位来源，不复制受版权保护的全文。
- `--project <project_id>`：包含用户个人信息、未公开项目或公司、私人对话、内部文件或专有数据的案例，写入 `private/projects/<project_id>/evidence/cases/`。
- `--public-example`：纯虚构示例，写入 `08-Examples/cases/`，并标记 `example: true` 与 `fictional: true`。

资料能在公开网页上看到，不代表可以无条件收录其中的个人数据。公开案例仍须满足目的相关、数据最小化、来源条款与适用法律；无法确认时保留来源链接和必要摘要，不收录敏感细节。

## 仓库结构

```text
00-Start/       静态入口与人工索引
01-Models/      公共思维模型
02-Checklists/  可选检查清单
03-Templates/   模型、案例、私人工作区模板
04-Prompts/     厂商无关的分析协议
05-Maps/        模型关系导航
06-Scripts/     初始化、创建与验证工具
07-Docs/        历史计划与研究文档
08-Examples/    仅限纯虚构的公开示例
09-Cases/       基于公开可核验来源的真实案例
.agents/        Codex 参考 Skill
private/        本地私人认知资产，Git 忽略
```

`PROJECT_SPEC.md` 定义公共产品与数据契约；`04-Prompts/LollaNet Analysis Protocol.md` 定义当前通用分析流程。`07-Docs/` 中标记为 `superseded` 的旧计划只保留为历史记录。

## 验证与研究

```bash
pytest -q
python 06-Scripts/validate_vault.py
python 06-Scripts/validate_vault.py --include-private
```

第一阶段另维护 10 个盲测任务，对比普通 AI 与 LollaNet，并按贴合度、非显然增量、简洁性、证据质量、行动价值和“此刻的优雅”成对排序。至少 7/10 次获选且事实可靠性不低于基线，才算对短期价值的初步支持；短视假设必须等待 30/90 天复盘。

## 常用工具

```bash
python 06-Scripts/create_model.py "模型名称" --discipline Psychology --proposal mp-accepted-proposal-id
python 06-Scripts/create_case.py --help
python 06-Scripts/create_project.py --help
python 06-Scripts/create_decision.py --help
python 06-Scripts/validate_vault.py --help
```
