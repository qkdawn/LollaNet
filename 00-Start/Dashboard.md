---
type: dashboard
title: LollaNet Dashboard
updated: 2026-10-09
---

# LollaNet 模型库

总结模型，说明机制与边界，保留来源，便于查找和比较。

## 浏览与查询

- [[Model Index]]：既有模型分类导航，属于冻结的 legacy 快照。
- [[Model Relationship Index]]：既有模型关系与学科分布导航。
- [[Model Graph]]：切换时保留的关系快照。
- [数据库文档](../db/README.md)：当前模型、分类和正式关系的查询方式。

正式内容以 PostgreSQL `lollanet` schema 为准，静态导航不代表数据库最新状态。

```powershell
python 06-Scripts/model_db.py search "基础概率"
python 06-Scripts/model_db.py neighbors mm-base-rate@3
python 06-Scripts/model_db.py traverse mm-base-rate@3 --max-depth 3
```

命令在仓库根目录执行，需配置 `LOLLANET_DATABASE_URL`。

## 模型整理标准

- 定义明确：说清模型解释的对象或问题。
- 机制清楚：说明关键变量及其关系。
- 边界明确：列出前提、适用条件、误用情境和反例。
- 来源可查：区分原始观点、研究证据与整理者推断。
- 组织有效：分类与关系帮助查找和比较，避免重复模型。

## 维护与验证

- [项目规格](../PROJECT_SPEC.md)：模型库职责与解耦边界。
- [代理规则](../AGENTS.md)：隐私、来源和正式内容维护要求。
- [数据库架构决策](../07-Docs/architecture-decisions/0001-postgresql-model-library.md)：存储与版本规则。

## 历史兼容内容

分析协议、私人认知层、项目与决策记录、分析 Skill 和研究实验仍保留在原目录。它们不属于当前模型库主流程，维护模型库无需初始化私人工作区或运行任务分析。
