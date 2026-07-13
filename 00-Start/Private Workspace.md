---
type: public_note
title: Private Workspace
updated: 2026-07-12
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
private/me/Long-term Goals.md
private/me/Principles.md
```

初始化还会创建：

```text
private/models/
private/evaluations/
```

身份记录事实性的“我是谁”，准则记录规范性的“我愿意怎样取舍”，二者都不等同于描述世界机制的思维模型。

AI 只读取会显著改变当前判断的最小必要内容。真实任务默认留在原环境；用户选择保存的身份、决策、评估、模型修改提案，以及包含用户个人信息、未公开项目或公司、私人对话、内部文件或专有数据的案例，必须位于 `private/`。

“真实”本身不是私有判据。完全基于公开可核验来源的真实案例可以进入 `09-Cases/<domain>/`，但必须记录非空 `source_refs`、`as_of`、事实与推断、时代条件，并遵守个人数据最小化、合法性和版权引用要求。纯虚构示例另放在 `08-Examples/cases/`。

模型提案使用：

```text
private/model-proposals/proposed/
private/model-proposals/accepted/
private/model-proposals/rejected/
private/model-proposals/deferred/
```
