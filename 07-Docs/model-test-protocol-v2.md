---
type: evaluation_protocol
title: LollaNet Model Test Protocol v2
updated: 2026-07-16
status: active
---

# LollaNet Model Test Protocol v2

本协议测试的是四个彼此不同的问题：模型卡是否具备可测试的质量、LollaNet 工作流是否带来整体增量、某张模型是否带来独立的行动增量，以及用户选择保存的真实任务是否在较长时间尺度上得到支持。它不把 88 张模型压缩为单一排名，也不以调用次数、短期成功率或固定加权总分自动改变模型状态；任何测试都不得自动改变模型状态。

本协议是 `07-Docs/evaluation-protocol.md` 中 v1 系统盲测的后续方法，不改写其历史结果。模型调用的保存、30/90 天复盘和 proposal 边界仍受 [Model Evolution Protocol](../04-Prompts/Model%20Evolution%20Protocol.md) 约束；正式变更仍只能通过 `private/model-proposals/` 的人工治理流程完成。

## 1. 进入条件与产物边界

- 测试应由用户或维护者明确启动；不因日常调用自动记录、提醒或汇总。
- 公共基准只可使用完全虚构、可公开的任务；真实 prompt、答案、评分、项目材料和私人上下文只能在用户明确保存后留在 `private/evaluations/` 或 `private/projects/<project_id>/reviews/`。
- 每轮开始前，将已填写的测试章程保存到私有目录并冻结任务范围、条件、评审规则、排除规则和报告方式；模板见 [model_test_charter_template.md](../03-Templates/model_test_charter_template.md)。
- 每轮只报告其实际测试的层级，不把系统层结果写成单卡结论，也不把私人任务结果写成公共模型事实。

## 2. 四个测试层级

| 层级 | 要回答的问题 | 主要方法 | 不能推出什么 |
| --- | --- | --- | --- |
| 卡片质量 | 这张卡是否有可解释、可回引且有边界的独立机制？ | 静态校验 + 人工语义审查 | 在真实决策中必然有效 |
| 系统增量 | 使用 LollaNet 是否优于相同条件下的普通回答？ | 盲法 pairwise：Baseline vs Full | 某张指定模型独自造成增益 |
| 模型独立增量 | 移除某张卡后，假设、预测、风险或行动是否有实质变化？ | 消融：Baseline / Full / Ablation | 模型在所有场景中都适用 |
| 长期适用性 | 在用户选择保存的真实任务中，预测和行动是否经得起复盘？ | 最小私人证据 + 即时/30/90 天复盘 | 私人结果自动修订公共模型 |

### 2.1 卡片质量门槛

进入任务级测试前，检查：

- 定义、机制、来源、反证/边界和可观察预测是否可回引；
- 形式化卡的符号、公式或算法、成立前提和失效条件是否齐全；
- 是否与现有卡在机制、变量、预测和行动含义上高度重复；
- `active` 卡是否达到更高的来源和边界质量；
- 是否存在应显式研究的冲突，而不是静默混合；
- 对 `formalism: true` 卡运行[形式化窄关系回归检查](formalism-regression-checks.md)，以防公式、算法或边界标记与可复算最小示例脱节；该检查不是来源正文复核、独立复现或现实有效性证据。

`python 06-Scripts/validate_vault.py` 只验证结构和契约；语义正确性、来源是否真的支持机制以及跨域转译是否成立，仍需要人工审查。

## 3. v2 首轮范围

### 3.1 系统盲测

使用 [benchmark-v2-task-matrix.md](../08-Examples/evaluation/benchmark-v2-task-matrix.md) 的 24 个完全虚构任务。它覆盖工程韧性、运营与资源配置、研究与解释、组织激励、长期选择和风险治理。所有 24 题都应完整报告，不能丢弃失利、平局或无效运行。

### 3.2 模型级消融：第一波

先测试四张刚完成形式化升级的工程学 active 卡，每张跨 3 个不同任务：

- `mm-margin-of-safety`（安全边际）
- `mm-redundancy`（冗余）
- `mm-bottleneck`（瓶颈）
- `mm-feedback-loop`（反馈回路）

具体的 12 个任务、预期的可区分增量和反例检查见 [engineering-core-ablation-cases-v2.md](../08-Examples/evaluation/engineering-core-ablation-cases-v2.md)。

后续波次可以从 active 卡中选择路径依赖、反向思考、能力圈、代理问题、激励导致的偏见、基础概率、贝叶斯更新和机会成本；是否扩大范围取决于首轮失败模式和人工复盘，而不是模型数量目标。

## 4. 系统级盲法：Baseline vs Full

1. 对每个任务运行一次 `Baseline`：不读取 LollaNet 文件、不使用 LollaNet 分析说明。
2. 以相同模型、相同任务文字、相同外部证据、相同工具权限和相同答案预算运行一次 `Full`：遵守分析协议，并只选 3–7 张真正改变假设、预测、风险或行动的非重复模型。
3. 公共虚构基准不读取私人资料；若未来测私人适配，必须在章程中说明用户授权的最小上下文、保持两条件可比，并避免在公开材料中复制它。
4. 将输出匿名随机标记为 `A` 与 `B`，由不知道来源的评审者独立评分。最好使用两位评审者；保留分歧，不以强制平均抹掉分歧。
5. 用 [blind_evaluation_v2_record_template.md](../03-Templates/blind_evaluation_v2_record_template.md) 保存完整记录。实际回答、映射表和评分只写入私有目录。
6. 全量报告胜/负/平、无效运行、评审分歧和代表性失败样例。分数是诊断信号，不是模型状态的自动触发器。

评审应逐项判断：任务贴合、独立且非显然的增量、证据质量、事实可靠性、行动价值、简洁性，以及是否明确了关键不确定性。更长、更花哨或出现更多模型名称的回答不得自动得分更高。

## 5. 模型级消融：Baseline / Full / Ablation

消融测试的唯一问题是：**目标模型是否给当前任务带来不可由其余已声明模型替代的行动增量？**

对每个目标模型—任务单元运行三种条件：

```text
Baseline  = 不使用 LollaNet
Full      = 预先声明的 3–7 张模型，包含目标模型
Ablation  = 与 Full 完全相同的已声明模型集合，但移除目标模型
```

执行要求：

- 在运行前记录目标模型 ID、版本、精确文件路径、Full 模型集合、任务、证据、工具权限和答案预算。
- Ablation 不得加入新的模型来重建目标模型的机制；也不得把目标模型换名引用或由提示词暗示其结论。
- 若目标模型与另一张卡机制重叠，应把这件事作为 `merge` 或 `contest` 的研究线索，而不是通过偷偷替换掩盖消融。
- 输出匿名后，评审优先检查：关键约束是否消失、可证伪预测是否变化、风险或反证是否遗漏、行动阈值/优先级/资源配置是否改变。
- “没有明显差异”是有效结果。它可能意味着不适用、被覆盖、调用方式装饰性，或该卡需要更清晰的行动转译；不能直接等同于模型错误。
- 使用 [model_ablation_record_template.md](../03-Templates/model_ablation_record_template.md) 保存结果。

## 6. 真实任务与延迟复盘

只有用户明确要求保存、复盘或检验时，才进入真实任务测试。使用 [model_usage_evidence_template.md](../03-Templates/model_usage_evidence_template.md) 记录最小必要材料：模型怎样改变假设、预测或行动；支持与反证；可观察信号；即时、30 天、90 天复盘；以及模型机制、适用性、执行、环境和随机性的归因不确定性。

在复盘窗口结束前，只能报告时间尺度覆盖、条件性预测和风险识别，不能宣称模型已经改善长期结果。私人调用证据可形成私人适配线索、继续研究理由或 proposal 建议；公共 `amend`、`merge`、`contest` 或 `deprecate` 仍需要公开、独立、可核验的证据。

## 7. 组合复盘与处置

在一批已保存的重要记录后，使用 [Model Portfolio Review Checklist](../02-Checklists/Model%20Portfolio%20Review%20Checklist.md) 审查重复、冲突、低频高损失风险覆盖、预测区分度、长期结果和私人/公共边界。

组合复盘只能建议：

- 维持；
- 继续观察；
- 继续研究；
- 展示 `new`、`amend`、`split`、`merge`、`contest` 或 `deprecate` proposal 草案。

任何文件写入、proposal 创建、proposal 接受和模型正式修订仍是分离步骤，均须获得相应的用户批准。
