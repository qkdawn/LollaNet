---
type: research_method_spec
title: RGSE：残差门控语义演化方法规格
status: draft
updated: 2026-07-18
---

# RGSE：残差门控语义演化方法规格

> 第二阶段产物：把 Proposal 中的 RGSE 从研究直觉收敛为可实现、可比较、可证伪的方法。

## 1. 方法目标与边界

RGSE 只解决一个核心问题：

> **在固定显式模型语义下，判断当前失败是否还有足够的路由改进空间；如果没有，判断是否应当让 LLM 提出模型语义候选。**

RGSE 不负责：

- 自动改变用户目标、价值函数、原则或安全策略；
- 直接从一次失败创造正式模型；
- 让 LLM 直接决定候选是否正确；
- 证明候选发现了新的科学定律；
- 在第一篇论文中实现无限制在线自我重写。

第一篇论文采用离线、版本化、可回滚的语义演化。在线环境只产生 Episode 和延迟结果，不直接覆盖正式模型库。

## 2. 冻结的研究问题与假设

### 2.1 研究问题

**RQ1：** 在固定模型语义和相同计算预算下，结果训练的多步模型路由是否比固定人工组合、语义检索、自由 LLM 路由和普通 Top-k 路由更能泛化到未见机制组合？

**RQ2：** 在固定语义模型库上，受限路由优化和反事实干预能否区分路由不足与语义缺失？

**RQ3：** 在路由改进空间不足且结构性残差持续出现时，RGSE 是否比无门控的 LLM 修改更可靠地生成可复用模型语义候选？

**RQ4：** 被接受的语义候选是否在未参与候选生成的任务、机制组合和时间段中继续产生增量价值，并且能通过节点和边干预验证其真实作用？

### 2.2 研究假设

- **H1：** 结果训练的多步路由在组合分布外测试上的结果损失和校准误差优于固定或自由路由基线。
- **H2：** RGSE 比随机触发、周期触发和无门控修改产生更少的无效语义更新、回归和复杂度增长。
- **H3：** RGSE 接受的候选在未见任务上仍有正的增量价值，而不是只改善候选生成集。
- **H4：** 删除、替换、打乱语义契约和强制改变顺序，会对相关预测、置信度或行动造成可预期变化。
- **H5：** 候选冻结后的独立接受门能降低直接接受 LLM 候选的过拟合和错误演化风险。

H2 和 H3 是 RGSE 的主要算法假设；H4 是显式模型不是装饰的必要条件；H1 是固定语义路由的前置结果。

## 3. 状态与数据对象

### 3.1 模型网络状态

```text
NetworkState
├── model_contracts: {Phi_i}
├── numeric_parameters: Theta
├── graph: E
├── route_policy: pi_Theta
├── version
├── safety_constraints
└── rollback_point
```

模型契约 `Phi_i` 至少包括：

```text
model_id
mechanism
inputs
outputs
assumptions
applicability
failure_conditions
falsifiable_predictions
evidence_refs
version
```

### 3.2 Episode

每个 Episode 记录：

```text
Episode
├── context x
├── goal G
├── observations o[0:T]
├── constraints K
├── activated_capabilities
├── activated_models
├── route_trace
├── model_outputs
├── pre-registered_predictions
├── confidence / calibration data
├── action
├── delayed_outcome
├── data_quality_flags
├── execution_quality_flags
└── residual_signature
```

候选生成器可以访问训练集、开发集和已授权的历史 Episode，但不能访问候选冻结前的封闭保留集或最终时间外测试集。

## 4. 高频回路：数值路由学习

路由器输入情境、目标、共享状态和候选模型契约的结构化表示，输出下一步调用：

```text
(C_k, M_i, alpha_i, stop)
```

其中 `C_k` 是能力算子，`M_i` 是显式模型，`alpha_i` 是调用强度或权重。

路由器可以是神经策略网络：

```text
pi_Theta(next_call | x, G, h, active_models, residual_summary)
```

第一版实现不要求所有模型节点可微。对于离散调用顺序，使用带轨迹回报的 contextual policy-gradient 或 contextual bandit；对于可微的合成模型，可以额外报告反向传播版本，但不能用不同的优化器给不同方法制造不公平优势。

单个 Episode 的训练目标可写成：

$$
\mathcal{L}_{route}
=\mathcal{L}_{result}
+\lambda_{cal}\mathcal{L}_{calibration}
+\lambda_{action}\mathcal{L}_{action}
+\lambda_{sparse}\Omega_{route}
+\lambda_{cost}\Omega_{compute}
+\lambda_{stable}\Omega_{update}
$$

其中：

- `L_result` 衡量预测或状态转移结果；
- `L_calibration` 衡量概率和置信度校准；
- `L_action` 衡量行动质量或风险调整后的效用；
- `Omega_route` 防止所有模型同时激活；
- `Omega_compute` 约束调用成本；
- `Omega_update` 防止最近一次 Episode 造成剧烈路由变化。

## 5. 路由不足与语义残差的责任分解

### 5.1 当前损失与路由上限

设 `B(Phi, Theta)` 是当前模型网络，`L_current` 是其在开发窗口上的损失。固定语义和模型库，只允许在预先定义的路由更新集合 `U_route` 中优化数值参数：

$$
\widehat{L}_{route}^{*}
=\min_{\Delta\Theta\in U_{route}}
\widehat{L}(B(\Phi,\Theta+\Delta\Theta))
$$

路由改进空间估计为：

$$
\widehat{H}_{route}
=\widehat{L}_{current}-\widehat{L}_{route}^{*}
$$

解释：

- `H_route` 较大：当前错误可能主要是路由问题，应继续训练 `Theta`；
- `H_route` 较小：固定语义下已经没有太多路由可改善空间，才有资格进一步检查语义残差。

`L_route*` 不是任意训练到最好，而是由预注册的计算预算、优化轮数、随机种子和停止条件定义。否则“路由已经充分优化”无法复现。

### 5.2 结构性残差的稳定性

对每个 Episode 计算标准化残差：

$$
q_e=\ell_e-\operatorname{E}[\ell\mid x_e,G_e,\text{available data}]
$$

残差不能只按数值大小触发。系统还要保留残差签名：

```text
residual_signature
├── prediction_direction
├── time_lag
├── violated_assumption
├── affected_output
├── mechanism_cluster
└── counterexamples
```

建议使用非重叠时间块计算稳定性。对每个候选残差簇 `r`，只有同时满足以下条件才标记为结构性残差：

1. 至少 `K` 个非重叠 Episode 块出现同方向的超额损失；
2. 每个块的样本数达到预设下限；
3. 通过预注册的 sequential / block-bootstrap 检验，超额损失的单侧置信下界高于 `delta_persist`；
4. 残差不能主要由数据质量或执行质量标记解释；
5. 该残差在至少两个任务或机制条件中复现，除非任务本身明确是单一环境研究。

这里的阈值不是由 LLM 选择，而是在开发集上预注册，并在封闭测试前冻结。

### 5.3 RGSE 触发门

定义三个门：

```text
G_data       = 数据质量与执行质量通过
G_route      = route_headroom 的上置信界 <= delta_route
G_persist    = 残差持久性检验通过
```

只有：

```text
Trigger_semantic_evolution
= G_data AND G_route AND G_persist
```

才允许向 LLM 发送语义演化请求。

这一步是 RGSE 的核心：LLM 不是每次失败都被调用，语义修改也不是路由训练失败后的默认补丁。

## 6. LLM 候选生成协议

LLM 接收：

- 当前模型契约及版本；
- 结构性残差样本；
- 事前预测与真实结果的差异；
- 反例和竞争性解释；
- 可用证据；
- 允许的操作类型；
- 不能访问的保留集说明。

LLM 只能输出结构化候选：

```json
{
  "operation": "amend | add | split",
  "parent_model_ids": [],
  "new_contract": {
    "mechanism": "",
    "inputs": [],
    "outputs": [],
    "assumptions": [],
    "applicability": [],
    "failure_conditions": [],
    "falsifiable_predictions": []
  },
  "why_existing_route_is_insufficient": "",
  "counterexamples": [],
  "provenance": []
}
```

候选生成后必须经过：

1. schema 校验；
2. 语义契约完整性校验；
3. 与已有模型去重和冲突检查；
4. 可计算性检查；
5. 预测是否可在实验环境中观察的检查。

这些检查只能过滤非法候选，不能替代最终结果接受门。

## 7. 候选接受、暂缓与回滚

### 7.1 候选冻结

候选一旦进入评估阶段，以下内容都必须冻结：

- LLM 版本和提示模板；
- 输入的残差集和证据集；
- 候选语义契约；
- 候选排序方法；
- 保留集和时间外测试集；
- 评价指标与接受阈值。

候选冻结后，不能通过修改提示、追加失败案例或人工改写候选来进行隐性调参。

### 7.2 接受指标

对每个候选 `j`，在相同路由预算下重新训练或校准 `Theta`，然后比较完整系统和基线系统：

$$
\Delta_j
=\widehat{L}_{base}^{val}
-\widehat{L}_{candidate,j}^{val}
-\lambda_{complex}\Delta Complexity_j
$$

候选必须满足：

1. 在封闭保留集上 `Delta_j` 高于预注册的最小增量 `delta_gain`；
2. 主要结果指标的单侧置信区间不跨越零，或通过预注册的层级/配对检验；
3. 校准、风险和安全约束没有超过允许退化；
4. 在至少一个未参与生成的任务或机制组合上有正增量；
5. 没有显著增加无效调用、模型数量或计算成本；
6. 多个候选竞争时通过 Holm 或 max-statistic 等预注册的多重比较校正。

“看起来合理”“解释更长”“LLM 置信度更高”都不能作为接受依据。

### 7.3 状态机

```text
proposed
  -> shadow
  -> accepted
  -> deployed

proposed -> rejected
proposed -> deferred
shadow   -> rolled_back
accepted -> deprecated
```

- `proposed`：LLM 生成但尚未验证；
- `shadow`：候选在不影响正式结果的影子环境中运行；
- `accepted`：通过封闭评估并进入候选模型库；
- `deployed`：在预先定义的安全范围内启用；
- `deferred`：证据不足但值得继续观察；
- `rejected`：未通过接受门；
- `rolled_back`：部署后触发回归或安全门；
- `deprecated`：历史上曾接受但后来被证明不再有用。

## 8. RGSE 算法伪代码

```text
Algorithm RGSE
Input:
    initial contracts Phi_0
    initial numeric parameters Theta_0
    training episodes D_train
    development episodes D_dev
    sealed episodes D_sealed
    out-of-time episodes D_oot
    thresholds and budgets B

State:
    Phi <- Phi_0
    Theta <- Theta_0
    version <- 0

For each update window W in chronological order:

    1. Run the current network B(Phi, Theta) on W.
       Record predictions, route traces, actions and delayed outcomes.

    2. Update Theta using only the declared route-training objective.
       Re-evaluate current loss L_current on D_dev.

    3. Estimate fixed-semantic route optimum:
       L_route_star <- bounded_route_optimization(Phi, W, B.route_budget)
       H_route <- L_current - L_route_star

    4. Estimate data and execution quality.
       If quality gates fail:
           record diagnostic state
           do not trigger semantic evolution
           continue numeric training or quarantine the window

    5. Update residual clusters and persistence statistics.
       If route headroom is not below delta_route:
           continue numeric training
       If residual persistence is not above delta_persist:
           continue numeric training

    6. Build a residual packet from train/dev data only.
       Ask the LLM for amend/add/split candidates.
       Validate schema, contract completeness and computability.
       Deduplicate candidates.
       Freeze candidate set C.

    7. For each candidate c in C:
           train/calibrate Theta_c under the same budget
           evaluate on D_sealed and D_oot
           compute gain, calibration, complexity and safety statistics

    8. Apply multiplicity correction and acceptance rules.
       If no candidate passes:
           keep Phi unchanged
           mark candidates rejected or deferred
       Else:
           select the simplest candidate with stable positive gain
           deploy first in shadow mode
           if shadow monitoring passes:
               accept Phi <- Phi_c
               increment version
           else:
               rollback

Return versioned history of Phi, Theta, routes, candidates and decisions.
```

伪代码中的关键约束是：候选生成、候选评估和候选接受不能共享同一封闭数据；路由优化预算必须预先固定；新模型的收益必须和重新训练 `Theta` 的收益分开报告。

## 9. 最小可行实验协议

第一版不要同时实现全部跨学科模型和金融系统。最小实验只需要：

1. 8–12 个机制模型；
2. 4–6 个合成环境；
3. 训练、开发、封闭保留和时间外四种数据分区；
4. 一个神经路由器；
5. 一个简单的 Episode 结果记录器；
6. 三种语义演化操作：`amend`、`add`、`split`；
7. 4 个语义演化对照：无演化、随机触发、无门控直接接受、完整 RGSE。

第一版先不需要：

- 全部七类能力都实现为独立神经网络；
- 无限模型增长；
- 复杂的多 Agent LLM 评审委员会；
- 真实交易执行；
- 用户私人数据；
- 在线自动修改正式公共模型。

最小实现的成功标准是：在固定语义路由任务上确认路由器确实学习；在人为缺失机制任务上确认残差触发和接受门可以被独立测量；在候选冻结后的保留集上确认结果而不是 LLM 自评决定候选去留。

## 10. 失败判据与停止规则

第二阶段之后，以下结果应触发研究方向调整：

- 普通路由在相同预算下已经达到完整 RGSE 的结果；
- `H_route` 估计不稳定，无法重现“路由充分”；
- 残差持久性只来自单一任务或单一数据集；
- LLM 候选无法编译为可计算的模型契约；
- 接受门对错误候选没有实际筛选作用；
- 新模型只在候选生成集有效；
- 语义契约干预不改变模型输出；
- 计算成本随模型数量增长而没有可接受的稀疏性控制。

若失败，应将论文改写为“显式模型路由基准/系统论文”，而不是继续声称 RGSE 是成功的语义演化算法。

## 11. 下一步

第二阶段完成后的下一步是实现最小机制环境和固定语义路由器，而不是继续扩展 Proposal 文本：

```text
方法规格冻结
    -> 编写 synthetic mechanism environment
    -> 编写 fixed-Phi route baseline
    -> 编写 route-headroom estimator
    -> 编写 residual persistence detector
    -> 编写 candidate evaluator
    -> 运行小规模先导实验
```

如果先导实验无法证明固定语义下的路由学习和残差责任分解，应该在进入 LLM 语义演化前停止扩展。

## 与 Proposal 的关系

本文件是 Proposal 的方法规格补充。Proposal 负责说明研究动机、论文边界和贡献主张；本文件负责给出 RGSE 的状态、触发门、候选协议、接受规则和最小实现。两者不应出现不同的阈值、数据切分或候选状态定义。
