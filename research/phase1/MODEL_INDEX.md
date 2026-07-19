---
type: research_model_index
scope: phase1_independent_assets
generated_by: research/phase1/generate_index.py
---

# Phase 1 Model Index

由 `generate_index.py` 自动生成；仅供浏览，不是训练结果。

## 数学

| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |
| --- | --- | --- | --- | --- |
| 函数与映射 (`mm-p1-function-mapping`) | reasoning | observation, transition | numeric | mm-p1-optimization-constraints, mm-p1-numerical-error |
| 局部敏感度 (`mm-p1-local-sensitivity`) | reasoning | observation, evaluation | numeric | mm-p1-numerical-error, mm-p1-safety-margin |
| 数值误差传播 (`mm-p1-numerical-error`) | reasoning | observation, challenge | numeric | mm-p1-local-sensitivity, mm-p1-safety-margin |
| 优化与约束 (`mm-p1-optimization-constraints`) | action | evaluation, control | categorical, numeric | mm-p1-local-sensitivity, mm-p1-opportunity-cost |

## 物理

| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |
| --- | --- | --- | --- | --- |
| 扩散与梯度 (`mm-p1-diffusion-gradient`) | world | transition, observation | time_series | mm-p1-feedback-loop, mm-p1-stability-transition |
| 能量守恒 (`mm-p1-energy-conservation`) | world | transition, challenge | time_series | mm-p1-force-balance, mm-p1-stability-transition |
| 力与平衡 (`mm-p1-force-balance`) | world | transition, control | numeric | mm-p1-energy-conservation, mm-p1-safety-margin |
| 稳定性与相变 (`mm-p1-stability-transition`) | world | transition, challenge | time_series | mm-p1-feedback-loop, mm-p1-safety-margin |

## 化学

| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |
| --- | --- | --- | --- | --- |
| 缓冲与容量 (`mm-p1-buffer-capacity`) | world | transition, control | numeric | mm-p1-safety-margin, mm-p1-feedback-loop |
| 平衡扰动响应 (`mm-p1-equilibrium-response`) | world | transition, challenge | categorical | mm-p1-buffer-capacity, mm-p1-feedback-loop |
| 物质守恒与限制试剂 (`mm-p1-mass-stoichiometry`) | world | transition, observation | numeric | mm-p1-rate-control, mm-p1-bottleneck |
| 速率控制 (`mm-p1-rate-control`) | world | transition, control | categorical | mm-p1-bottleneck, mm-p1-equilibrium-response |

## 工程

| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |
| --- | --- | --- | --- | --- |
| 瓶颈 (`mm-p1-bottleneck`) | world | observation, control | time_series | mm-p1-rate-control, mm-p1-reliability-lifetime |
| 反馈回路 (`mm-p1-feedback-loop`) | world | transition, challenge | time_series | mm-p1-diffusion-gradient, mm-p1-buffer-capacity |
| 可靠度与寿命分布 (`mm-p1-reliability-lifetime`) | world | observation, transition, control | categorical | mm-p1-bottleneck, mm-p1-safety-margin |
| 安全边际 (`mm-p1-safety-margin`) | action | evaluation, control, challenge | probability | mm-p1-reliability-lifetime, mm-p1-expected-value |

## 金融

| 模型 | Kind | 职责 | 关键输入类型 | 可组合模型 |
| --- | --- | --- | --- | --- |
| 贝叶斯更新 (`mm-p1-bayesian-update`) | reasoning | observation, evaluation, challenge | probability | mm-p1-expected-value, mm-p1-safety-margin |
| 复利 (`mm-p1-compounding`) | world | transition, evaluation | numeric | mm-p1-opportunity-cost, mm-p1-safety-margin |
| 期望值 (`mm-p1-expected-value`) | action | evaluation, control | categorical | mm-p1-opportunity-cost, mm-p1-safety-margin |
| 机会成本 (`mm-p1-opportunity-cost`) | action | evaluation, challenge | numeric | mm-p1-optimization-constraints, mm-p1-compounding |

## 组合关系

- `mm-p1-bayesian-update`: mm-p1-bayesian-update -> mm-p1-expected-value。先更新情景概率，再比较行动的期望结果
- `mm-p1-bottleneck`: mm-p1-bottleneck -> mm-p1-safety-margin。缓解约束时同时检查可用容量与失效风险
- `mm-p1-buffer-capacity`: mm-p1-buffer-capacity -> mm-p1-safety-margin。将剩余缓冲量转化为可接受风险边界
- `mm-p1-compounding`: mm-p1-compounding -> mm-p1-opportunity-cost。比较不同路径在长期累积后的放弃成本
- `mm-p1-diffusion-gradient`: mm-p1-diffusion-gradient -> mm-p1-feedback-loop。将传播延迟纳入反馈系统的动态判断
- `mm-p1-energy-conservation`: mm-p1-energy-conservation -> mm-p1-stability-transition。由能量与扰动判断稳定状态是否可能改变
- `mm-p1-equilibrium-response`: mm-p1-equilibrium-response -> mm-p1-buffer-capacity。判断扰动后系统能否维持关键状态范围
- `mm-p1-expected-value`: mm-p1-expected-value -> mm-p1-opportunity-cost。在预计收益之外计入放弃最佳替代方案的代价
- `mm-p1-feedback-loop`: mm-p1-bottleneck -> mm-p1-feedback-loop。判断缓解约束的行动会否经延迟反馈制造新的拥塞
- `mm-p1-force-balance`: mm-p1-force-balance -> mm-p1-safety-margin。先求工作载荷，再判断是否接近失效边界
- `mm-p1-function-mapping`: mm-p1-function-mapping -> mm-p1-optimization-constraints。先定义变量关系，再在关系上求可行最优解
- `mm-p1-local-sensitivity`: mm-p1-optimization-constraints -> mm-p1-local-sensitivity。用敏感度检查最优解对关键参数是否脆弱
- `mm-p1-mass-stoichiometry`: mm-p1-mass-stoichiometry -> mm-p1-rate-control。先确定可达产出上限，再判断达到上限的速度
- `mm-p1-numerical-error`: mm-p1-numerical-error -> mm-p1-safety-margin。用不确定性预算确定必要的安全缓冲
- `mm-p1-opportunity-cost`: mm-p1-optimization-constraints -> mm-p1-opportunity-cost。在资源约束下比较不同可行方案的放弃代价
- `mm-p1-optimization-constraints`: mm-p1-optimization-constraints -> mm-p1-opportunity-cost。在可行解中比较放弃不同方案的代价
- `mm-p1-rate-control`: mm-p1-rate-control -> mm-p1-bottleneck。比较化学控速步骤与系统吞吐约束是否为同一限制
- `mm-p1-reliability-lifetime`: mm-p1-reliability-lifetime -> mm-p1-bottleneck。判断关键约束环节的失效是否会主导总吞吐风险
- `mm-p1-safety-margin`: mm-p1-reliability-lifetime -> mm-p1-safety-margin。用故障风险与能力退化确定可接受裕度
- `mm-p1-stability-transition`: mm-p1-stability-transition -> mm-p1-safety-margin。在临界阈值前设定可接受扰动与缓冲
