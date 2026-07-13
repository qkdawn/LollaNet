---
type: dashboard
title: LollaNet Dashboard
updated: 2026-07-13
---

# LollaNet Dashboard

## 开始一次分析

- [[LollaNet Analysis Protocol]]：厂商无关的现行分析协议。
- Codex：使用 `$lollanet-analyze <当前任务>` 调用参考 Skill。
- 简单事实、格式转换或低风险执行任务可直接完成，不必启动多路召回。

真实任务默认留在当前聊天、仓库或应用中。完成任务不等于必须创建项目、决策或长期记录。

## 认知资产

- [[Model Index]]：浏览现有公共模型。
- [[Model Relationship Index]]：查看模型关系与学科分布。
- [Public Case Library](../09-Cases/README.md)：浏览基于公开可核验来源的真实案例。
- [[Private Workspace]]：了解本地身份、准则、提案和可选项目空间。
- `private/me/User Profile.md`：事实性的“我是谁”。
- `private/me/Principles.md`：规范性的“我愿意怎样取舍”。
- `private/models/`：私人模型补充、候选或适用边界。
- `private/model-proposals/`：由人审议的模型修改建议。
- `private/evaluations/`：盲评与 30/90 天延迟复盘。

模型应按当前问题的结构选择。一次分析只保留 3–7 个能改变假设、预测或行动的非重复模型，不设固定“高频模型”清单。

## 分析与复盘

- [[Analysis Quality Checklist]]：检查证据、替代解释、行动和不确定性。
- [[Decision Checklist]]：重大且需要完整检查时使用。
- [[Weekly Review]]：用户选择沉淀后再进行周期复盘。
- [[Inbox Processing Report]]：处理用户主动放入长期层的材料。
- [[PROJECT_SPEC]]：查看未来外部认知输入闭环的边界与数据契约。
- [External Input Loop Plan](../07-Docs/plans/external-input-loop.md)：RSS、内容转换与模型迭代意见日报的实验方案。
- [Daily Model Iteration Report Template](../03-Templates/daily-model-iteration-report_template.md)：私人日报模板。

## 可选沉淀

初始化私人层：

```bash
python 06-Scripts/init_private.py
```

只有用户希望长期保留时才创建项目、决策或私人案例；公开来源真实案例与虚构示例使用各自独立入口：

```bash
python 06-Scripts/create_project.py <project_id> --name "<项目名称>"
python 06-Scripts/create_decision.py "<决策标题>" --project <project_id>
python 06-Scripts/create_case.py "<私人案例标题>" --project <project_id>
python 06-Scripts/create_case.py "<公开案例标题>" --public-case --source "<可核验来源>"
python 06-Scripts/create_case.py "<虚构示例标题>" --public-example
```

三个案例目标互斥：公开来源真实案例进入 `09-Cases/<domain>/`，必须保留非空来源、`as_of`、事实/推断边界和时代条件；私人或未公开案例进入 `private/projects/<project_id>/evidence/cases/`；纯虚构示例进入 `08-Examples/cases/`，并标记 `example: true`、`fictional: true`。公开可见不等于个人数据可任意收录，仍须最小化并确认合法性；引用来源，不复制受版权保护的全文。

## 模型治理

- 模型状态：`candidate`、`active`、`contested`、`deprecated`。
- 修改类型：`new`、`amend`、`split`、`merge`、`contest`、`deprecate`。
- AI 默认只读；建议先展示，明确批准后才写入私人提案。
- 接受提案后才创建候选模型、递增版本或改变状态；废弃模型保留历史。

## 项目与验证

- [[PROJECT_SPEC]]：现行产品、数据和兼容性规格。
- `07-Docs/`：历史研究和旧计划；`superseded` 文档不可作为当前实现依据。

```bash
pytest -q
python 06-Scripts/validate_vault.py
python 06-Scripts/validate_vault.py --include-private
```

LollaNet 的三个目标仍是待验证假设。短期价值通过 10 个盲测任务比较，短视问题必须等待 30/90 天复盘，不能提前宣称已经改善。
