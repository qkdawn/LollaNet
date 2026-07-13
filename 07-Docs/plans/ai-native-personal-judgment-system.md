---
type: implementation_plan
status: superseded
superseded_on: 2026-07-12
superseded_by:
  - PROJECT_SPEC.md
  - 04-Prompts/LollaNet Analysis Protocol.md
---

# LollaNet AI-Native 个人判断系统生成器计划书

> **状态：已被取代。** 本文仅保留早期设计历史，不得继续执行。其中“所有真实决策必须入项目”等规则已经失效；现行产品与数据契约以 `PROJECT_SPEC.md` 为准，运行流程以 `04-Prompts/LollaNet Analysis Protocol.md` 为准。

## Summary

将 LollaNet 升级为一个可共享的 AI 判断系统框架：公共仓库只保留模型、模板、协议、脚本和说明；每个使用者的用户画像、项目、决策日志、证据和复盘都由 AI 在本地 `private/` 中生成，并默认不提交。

核心抽象：

```text
LollaNet = public framework + private identity + isolated projects
```

## Key Changes

- 新增 `AGENTS.md` 作为 AI 操作协议。
- 新增 `.gitignore`，忽略 `private/`、缓存和 Obsidian workspace 状态文件。
- 新增 `06-Scripts/init_private.py`，初始化本地私有身份和 `inbox` 项目。
- 新增 `06-Scripts/create_project.py`，生成隔离项目空间。
- 修改 `06-Scripts/create_decision.py`，真实决策默认必须写入 `private/projects/<project_id>/`。
- 新增用户画像、长期目标、项目画像和项目 Dashboard 模板。
- 调整 README、Dashboard 和分析协议，明确公共层和私有层边界。

## AI Workflow

1. 读取 `AGENTS.md` 和 `README.md`。
2. 如果没有 `private/`，运行 `python 06-Scripts/init_private.py`。
3. 判断用户问题属于哪个项目。
4. 没有合适项目时，运行 `python 06-Scripts/create_project.py <project_id> --name "<项目名称>"`。
5. 分析前读取 `private/me/User Profile.md`、`private/projects/<project_id>/Project Profile.md` 和 `04-Prompts/LollaNet Analysis Protocol.md`。
6. 调用 LollaNet 模型分析问题。
7. 需要沉淀决策时，运行 `python 06-Scripts/create_decision.py "<决策标题>" --project <project_id>`。
8. 证据写入 `private/projects/<project_id>/evidence/`。
9. 复盘写入 `private/projects/<project_id>/reviews/`。

## Test Plan

- `python 06-Scripts/init_private.py`
- `python 06-Scripts/create_project.py test-project --name "测试项目"`
- `python 06-Scripts/create_decision.py "测试决策" --project test-project`
- `python 06-Scripts/create_decision.py "测试决策"` should fail
- `python 06-Scripts/validate_vault.py`
- `python 06-Scripts/validate_vault.py --include-private`
- `pytest -q`

## Assumptions

- `private/` 放在仓库内，但永远不提交。
- 公共仓库不包含任何具体用户项目。
- `inbox` 是临时项目收集箱，不是默认真实项目。
- 所有真实决策必须进入某个 `private/projects/<project_id>/`。
- Obsidian 仍打开仓库根目录，同时使用公共框架和本地私有层。
