---
type: prompt
title: Model Card Prompt
updated: 2026-07-12
---

# Model Card Prompt

请把一个思维模型整理成 LollaNet 模型卡。目标不是解释概念，而是形成可被 AI 召回、质疑、验证和修订的判断工具。

## 演化约束

- 新模型一律以 `candidate` 创建；阅读、一次失败或单个案例只能产生修改提案，不能直接创建 `active` 模型。
- AI 只能建议 `new`、`amend`、`split`、`merge`、`contest` 或 `deprecate`，不能直接改写正式模型。
- 接受提案后才更新卡片；修改已有模型时递增整数 `version` 并记录修订依据。
- `deprecated` 模型保留原文件与修订历史，不删除；有替代模型时通过 `supersedes` 连接稳定 ID。
- 公共卡不得包含个人使用线索、真实项目、真实决策或使用记录；这些内容只进入 `private/`。

## Frontmatter

必须包含：

- `type: mental_model`
- 唯一且不随标题或路径变化的 ASCII `id`，格式为 `mm-*`
- `kind: world|reasoning|action`
- `status: candidate|active|contested|deprecated`
- 从 1 开始的整数 `version`
- `name`、`english_name`、`discipline`、`created`、`updated`
- `conflicts` 与 `supersedes`，没有关系时写 `[]`

`kind` 的含义：

- `world`：描述世界中的耐久机制。
- `reasoning`：提供分析或校准方法。
- `action`：提供可执行的行动或风险控制策略。

## 正文结构

严格使用 `03-Templates/mental_model_template.md` 的章节，至少包含：

- `## 一句话定义`
- `## 核心机制`
- `## 耐久机制`
- `## 时代性表现`
- `## 适用边界`
- `## AI 调用条件`
- `## 不应调用`
- `## 场景化决策提问`
- `## 典型误用`
- `## 反向思考`
- `## 经典案例`
- `## 支持证据`
- `## 反证`
- `## 失效条件`
- `## 可观察预测`
- `## 关联模型`
- `## AI 输出要求`
- `## 修订历史`

## 写作要求

- 区分耐久机制与时代性表现；历史类比必须比较机制、制度、技术和成本结构。
- `english_name` 不能为空；不知道正式英文名时写 `Needs English Name`。
- 不使用空白占位语；不确定时明确写“来源需要核验”或“暂无可靠证据”。
- 来源必须能回指书籍、论文、档案、财报、访谈或一手材料；模型解释本身不能充当证据。
- 支持证据与反证都要写证据类型、适用条件和局限，不能只列赞成或反对观点。
- 可观察预测必须允许未来证据降低模型置信度，不能写成永远正确的解释。
- 每个场景至少给 3 个能改变假设、预测或行动的问题。
- 关联模型必须区分“常一起调用 / 容易混淆 / 互相校正”；真正存在张力的模型另写入 `conflicts`。
