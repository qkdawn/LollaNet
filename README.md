# LollaNet

LollaNet 是一个跨学科模型库，专注于总结、整理和维护可复用的模型。

每个模型说明它解释什么、如何起作用、在什么条件下成立，以及何时不应使用。模型库的价值是内容准确、来源可追溯、边界清楚、便于查找和比较。

## 核心范围

- 模型：定义、核心机制、前提、适用边界、反例或反证、来源。
- 分类：按学科和机制组织模型，支持检索。
- 关系：记录模型之间的依赖、组合、竞争、修正和特化关系。
- 维护：去重、补充证据、修订内容、保留版本历史。

个人身份、价值、目标、项目记忆、任务分析、回答生成和训练实验属于外部使用场景，不是模型库的职责。使用者可以自行选择和组合模型；本库不规定必须使用几条模型或怎样完成一次分析。

## 模型存储与查询

沿用现有存储决策：PostgreSQL `lollanet` schema 是正式模型库的唯一事实来源。`01-Models/` 和静态索引是切换时保留的 legacy 快照，不应反向覆盖数据库。

```powershell
python -m pip install -r requirements.txt
$env:LOLLANET_DATABASE_URL = "postgresql://user:password@localhost:5432/lollanet"
python 06-Scripts/model_db.py migrate
python 06-Scripts/model_db.py import-models --root . --dry-run
python 06-Scripts/model_db.py import-models --root .
python 06-Scripts/model_db.py audit
python 06-Scripts/model_db.py search "基础概率"
python 06-Scripts/model_db.py neighbors mm-base-rate@3
python 06-Scripts/model_db.py traverse mm-base-rate@3 --max-depth 3
```

导入命令用于既有 Markdown 快照迁移，不是日常编辑入口。配置、事务与回滚说明见 [数据库文档](db/README.md)。

## 入口与边界

- [模型与曲线工作台](frontend/README.md)：Next.js + Go 本地只读前端，浏览 88 个模型的公开快照与 Behavior Mode 实验分类。
- [模型库导航](00-Start/Dashboard.md)
- [项目规格](PROJECT_SPEC.md)
- [代理维护规则](AGENTS.md)
- [数据库架构决策](07-Docs/architecture-decisions/0001-postgresql-model-library.md)

现有分析协议、Codex 分析 Skill、私人工作区、任务脚本和 `research/` 实验暂保留为历史兼容内容，不属于当前产品范围，也不是维护模型库的前置依赖。本次解耦明确职责与入口，尚未搬迁或删除这些文件。

## 验证

```powershell
python 06-Scripts/model_db.py import-models --root . --dry-run
python 06-Scripts/validate_vault.py
pytest -q tests/test_model_db_unit.py
```
