---
type: prompt
title: Case Analysis Prompt
updated: 2026-07-12
---

# Case Analysis Prompt

请把案例整理为 LollaNet 案例卡：区分来源直接支持的事实、来源作者的解释与 AI 推断，保留可精确回引的来源和 `as_of`，说明制度、技术、测量口径等时代条件，提炼失败机制或关键转折，并比较至少一个竞争性解释。连接模型时说明适用边界和反证，不能按表面相似套历史类比。

写入前按材料属性选择且仅选择一个目标：

- 公开来源真实案例：仅当底层来源公开且可核验时，使用 `--public-case --source "<ref>"` 写入 `09-Cases/<domain>/`。`source_refs` 必须非空；引用并概括必要内容，不复制受版权保护的全文。公开可见不等于个人数据可任意收录，仍须最小化并确认合法性。
- 私人案例：只要包含用户个人信息、未公开项目或公司、私人对话、内部文件或专有数据，就使用 `--project <project_id>` 写入 `private/projects/<project_id>/evidence/cases/`。
- 纯虚构示例：使用 `--public-example` 写入 `08-Examples/cases/`，并标记 `example: true`、`fictional: true`。不得把私人材料仅靠去标识化改称虚构示例。

案例可以生成模型修改提案，但不得直接修改公共模型。
