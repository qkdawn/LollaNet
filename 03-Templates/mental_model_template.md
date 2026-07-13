---
type: template
template_for: mental_model
updated: 2026-07-12
---

# {{name}}

## Required Frontmatter

创建真实模型卡时，用以下 schema 替换模板 frontmatter：

```yaml
---
type: mental_model
id: "{{id}}"
name: "{{name}}"
english_name: "{{english_name}}"
discipline: "{{discipline}}"
subdiscipline:
kind: reasoning
status: candidate
version: 1
importance: 3
frequency: medium
mastery: 1
created: "{{date}}"
updated: "{{date}}"
tags:
  - mental-model
  - decision-making
related: []
conflicts: []
supersedes: []
use_cases: []
---
```

## 一句话定义

一句话说明这个模型检查什么变量，以及它如何改变判断。

## 核心机制

说明因果链、关键变量，以及哪些条件会增强或削弱机制。

## 耐久机制

说明跨越时代仍可能成立的底层约束、行为规律或因果关系，不要把历史案例的表象当作机制本身。

## 时代性表现

说明制度、技术、成本结构和信息环境如何改变模型的表现形式与强度。

## 芒格式解释

把模型转为可执行的检查方法。不要伪造芒格原话；没有确定出处时明确写“来源需要核验”。

## 适用边界

- 适用：
- 不适用：
- 容易误判的边界：

## AI 调用条件

当问题出现以下信号时调用本模型：

-
-
-

## 不应调用

-
-
-

## 场景化决策提问

### 职业方向

1.
2.
3.

### 创业/产品

1.
2.
3.

### 投资/商业

1.
2.
3.

### 合作/组织

1.
2.
3.

## 典型误用

-
-
-

## 反向思考

如果这个模型被误用，最可能把判断带向哪个错误结论？

## 经典案例

### 案例事实


### 模型解释


### 来源

来源需要核验。

## 支持证据

- 支持机制的证据：
- 证据类型与强度：
- 仍需核验的部分：

## 反证

- 削弱该模型的观察：
- 可替代解释：
- 与模型冲突的案例：

## 失效条件

-
-
-

## 可观察预测

- 如果模型成立，应先观察到：
- 随后应观察到：
- 出现什么结果时应降低置信度：

## 关联模型

### 常一起调用


### 容易混淆


### 互相校正


## AI 输出要求

调用本模型后，输出必须包含：

- 关键变量
- 当前证据与反证
- 适用边界与最大不确定性
- 可观察预测
- 下一步验证动作

## 修订历史

| 日期 | 版本 | 状态 | 变更 | 依据或提案 |
| --- | --- | --- | --- | --- |
| {{date}} | 1 | candidate | 创建候选模型 | {{accepted_proposal_id}} |
