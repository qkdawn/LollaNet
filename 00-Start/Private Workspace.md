---
type: public_note
title: Private Workspace
updated: 2026-07-14
---

# Private Workspace

真实用户画像不存放在公共目录。

第一次使用 LollaNet 时运行：

```bash
python 06-Scripts/init_private.py
```

然后按需填写：

```text
private/me/User Profile.md
private/me/Value Function.md
private/me/Long-term Goals.md
private/me/Principles.md
private/me/Current Focus.md
private/me/dashboard.md
```

初始化还会创建：

```text
private/models/
private/evaluations/
```

个人层按语义分开：

- `User Profile.md` 记录事实性的“我是谁、拥有什么、受什么约束”；
- `Value Function.md` 记录“什么本身值得”、硬约束和未解决的价值冲突；
- `Principles.md` 记录“在这些条件下我愿意怎样行动”；
- `Long-term Goals.md` 记录服务价值、但可以替换的目标状态；
- `Current Focus.md` 记录当前 6–12 周的可撤回资源配置和实验；
- `observations/` 与 `reviews/` 保存证据，不能自动改写正式个人资产。

价值函数不是永久的全局加权公式。AI 可以提出候选表达，但不能从行为观察自动推导终极价值，也不能把候选条目升级为 `active`。

AI 只读取会显著改变当前判断的最小必要内容。真实任务默认留在原环境；用户选择保存的身份、决策、评估、模型修改提案，以及包含用户个人信息、未公开项目或公司、私人对话、内部文件或专有数据的案例，必须位于 `private/`。

“真实”本身不是私有判据。完全基于公开可核验来源的真实案例可以进入 `09-Cases/<domain>/`，但必须记录非空 `source_refs`、`as_of`、事实与推断、时代条件，并遵守个人数据最小化、合法性和版权引用要求。纯虚构示例另放在 `08-Examples/cases/`。

模型提案使用：

```text
private/model-proposals/proposed/
private/model-proposals/accepted/
private/model-proposals/rejected/
private/model-proposals/deferred/
```
