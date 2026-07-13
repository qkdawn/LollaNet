---
type: report
title: Inbox Processing Report
updated: 2026-07-12
---

# Inbox Processing Report

## 待处理

用户主动选择保留的临时材料可以放入 `private/projects/inbox/evidence/`。处理时先区分来源事实、来源作者的解释和 AI 推断，再判断它应继续作为证据、整理为私人案例、基于公开来源另建公共案例，还是生成 `private/model-proposals/proposed/` 下的候选提案。

单次阅读、观察或失败不得直接修改公共模型。原始材料不因整理而删除；包含用户信息、未公开项目或公司、私人对话、内部文件或专有数据的原始材料不得复制到公共目录。

只有底层资料本身公开且可核验时，才可在 `09-Cases/<domain>/` 新建真实案例；必须提供非空 `source_refs`、`as_of`，分开记录事实与推断并说明时代条件。只引用和概括必要内容，不复制受版权保护的全文。来源公开也不自动授权收录个人数据，仍须检查目的相关、最小化和合法性。纯虚构示例应从头创作并放入 `08-Examples/cases/`，不得把私人材料仅靠去标识化改称示例。

## 已处理记录

待补充。
