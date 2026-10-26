---
type: prompt
title: LollaNet Analysis Protocol
version: 5.0
updated: 2026-07-30
---

# LollaNet Analysis Protocol

LollaNet 用任务事实拼出两条可证伪的机制链，再比较哪条更符合观察。它不是模型清单。

## 最小流程

### 1. 事实

明确一个问题，并记录最少的带来源观察。每条只标记为 `fact`、`inference` 或 `unknown`。模型卡不能代替任务证据。

### 2. 两条机制链

对同一个问题构造至少两套程序。每套只有五项：

1. 一个可证伪主张；
2. 2–5 个模型及其角色；
3. 连接全部模型的关系；
4. 一个能区分对手的带时间预测；
5. 一个最小反证。

删除后不改变机制或预测的模型，不应进入程序。正式关系只有 `requires`、`composes`、`competes`、`corrects`、`specializes`；未注册关系只需说明一句 `why`。

### 3. 编译

```text
python 06-Scripts/model_graph.py --root . compile-context --input <pack.json-or->
```

Pack 只填写问题、观察和两套程序。编译器自动读取模型卡的核心机制，验证版本与关系，并生成实际调用图。`--input -` 表示不落盘。

### 4. 回答

自然回答三个问题：当前哪条解释更强、哪个对手仍成立、下一项什么观察能区分它们。末尾只附实际模型版本和关系。

## 成功标准

至少增加一条非显然机制链、竞争解释或区分预测。只增加模型名称、检查项或篇幅，等于失败。

## 默认不保存

默认不保存任务记录，不得为此额外读取私人资料，不得自动创建 proposal 或修改任何正式模型。保存与修订遵循 [Model Evolution Protocol](Model%20Evolution%20Protocol.md)。
