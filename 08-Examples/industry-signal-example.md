---
type: example
title: Fictional Industry Signal Example
updated: 2026-07-04
example: true
fictional: true
---

# Fictional Industry Signal Example

这是公共仓库中的纯虚构示例，用来展示行业信号记录格式。`08-Examples/` 不接收真实材料，即使材料已经去标识化。

包含用户信息、未公开项目或公司、私人对话、内部文件或专有数据的案例应写入：

```text
private/projects/<project_id>/evidence/
```

完全基于公开可核验来源的真实案例应写入 `09-Cases/<domain>/`，并保留非空来源、`as_of`、事实与推断边界及时代条件。

## Signal

- Source: Fictional market memo
- Date: 2099-01-01
- Reliability: low
- Claim: Example industry demand is increasing.

## Model Check

- [[基础概率]]：同类信号通常多大概率转化为真实需求？
- [[反馈回路]]：这个信号会不会吸引更多供给，反过来改变竞争格局？
- [[机会成本]]：如果追这个信号，会放弃什么替代方向？
