---
type: research_note
title: LollaNet v1：相关工作与贡献缺口矩阵
status: superseded
superseded_by: ../../trainable-cognitive-model-network.md
updated: 2026-07-28
---

# LollaNet v1：相关工作与贡献缺口矩阵

> 历史研究稿，保留作路线比较；不代表当前 LollaNet 运行时。

> 研究阶段：第一步——系统文献探索与贡献缺口判断  
> 检索截止：2026-07-18  
> 文档性质：Proposal 的研究依据，不是已完成的实验结果

## 1. 本轮要回答的问题

本轮不证明 LollaNet 已经有效，而是先判断它是否仍有一个足够清晰、可被算法化和实验验证的研究缺口。重点比较以下能力是否已经分别被已有工作覆盖：

1. 显式的、具有可解释语义的计算单元；
2. 学习选择哪些单元、以什么顺序和强度组合；
3. 根据真实执行结果而不是只根据文本合理性更新网络；
4. 允许语言模型提出持久模型的修改或新增；
5. 由独立的结果验证门决定候选是否正式进入系统；
6. 修改后的单元能否跨任务复用，并通过干预证明它真正参与了计算。

这里的“缺口”指在本轮纳入的代表性工作中尚未看到被同一系统同时形式化和验证的组合；它不等于证明全球文献中绝对没有相同方法。

## 2. 检索范围与纳入标准

### 2.1 检索范围

本轮覆盖六条相邻路线：

- neural module networks、modular continual learning 和 learned routing；
- concept bottleneck、language-guided concepts 和可编辑概念；
- LLM tool creation、tool retrieval、tool-use learning 和 tool verification；
- self-improving / self-evolving agents；
- LLM 假设生成与自动科学发现；
- 时间分布变化、时间外泛化和金融回测验证。

优先纳入论文官网、会议论文集、OpenReview、ACL Anthology、PMLR、CVF Open Access、Nature 和作者公开的原始论文页面。博客、新闻稿和二手综述只作为线索，不作为核心证据。

### 2.2 比较对象的定义

本文把 LollaNet 的目标对象定义为**显式跨学科模型节点**，而不是普通工具、提示词片段或隐藏神经模块。一个候选模型节点至少应有：

- 机制或因果职责；
- 结构化输入输出；
- 成立前提与适用边界；
- 可证伪预测或可检查约束；
- 版本、来源和证据记录。

目标网络学习的不是“哪个词最相关”，而是：

```text
给定情境和目标，学习哪些显式模型应当以什么顺序、强度和结构共同工作。
```

语义演化则限定为低频提案回路：

```text
稳定结构性残差
    -> 排除数据、执行和路由原因
    -> LLM 生成 amend / add / split 候选
    -> 候选冻结
    -> 历史回放 + 封闭保留集 + 时间外测试
    -> 接受、暂缓或回滚
```

## 3. 贡献维度

矩阵使用以下缩写：

- **S**：显式语义单元——单元的身份、职责或概念由人可读定义，而非只有隐向量；
- **R**：学习路由——学习选择、顺序、激活强度或模块间连接；
- **O**：结果反馈——使用执行后的真实标签、环境回报或长期结果更新系统；
- **E**：结构/语义演化——可以增加、修改、拆分或废弃持久单元；
- **L**：LLM 生成候选——语言模型参与候选工具、概念、假设或策略的生成；
- **G**：独立接受门——候选生成器与最终接受判据在机制上解耦；
- **X**：跨任务复用——修改后的单元作为持久资产在不同任务或环境中复用；
- **I**：机制干预证据——通过删除、替换、打乱顺序或强制路由证明单元确实改变了结果。

“部分”表示只覆盖相邻能力，不能直接视为等价实现。

## 4. 代表性相关工作矩阵

| 工作 | S | R | O | E | L | G | X | I | 与 LollaNet 的主要差异或缺口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Neural Module Networks，CVPR 2016 [1] | 部分 | 部分 | 否 | 否 | 否 | 否 | 部分 | 部分 | 将问题解析为模块组合，奠定显式模块路由思想；但模块语义和组合通常依赖任务解析或监督，未形成长期结果驱动的模型演化回路。 |
| Continual Learning via Local Module Composition，NeurIPS 2021 [2] | 否/部分 | 是 | 部分 | 否 | 否 | 否 | 是 | 部分 | 证明模块可在多任务和持续学习中组合、复用；核心单元仍是神经模块，不是带机制、边界和可证伪预测的跨学科模型。 |
| Scalable Modular Network，ICLR 2024 [3] | 否/部分 | 是 | 部分 | 否 | 否 | 否 | 是 | 部分 | 通过 agreement routing 解决模块选择和扩展问题；与 LollaNet 最接近的是路由学习，但没有 LLM 语义提案和独立的语义接受门。 |
| Concept Bottleneck Models，ICML 2020 [4] | 是 | 部分 | 部分 | 否 | 否 | 否 | 部分 | 是 | 把人类可理解概念放在预测中间层并支持干预；概念不是具有机制、适用边界和多步调用顺序的“模型节点”。 |
| LaBo，CVPR 2023 [5] | 是 | 部分 | 否 | 部分 | 是 | 部分 | 部分 | 是 | 使用语言模型从文本提出概念并构造可解释预测器，说明 LLM 可以参与语义单元生成；但目标是视觉概念发现，不是由长期结果驱动的跨学科模型网络演化。 |
| Language-Guided Concept Bottleneck Models，CVPR 2025 [6] | 是 | 部分 | 部分 | 部分 | 是 | 部分 | 部分 | 是 | 进一步展示语言引导概念空间和持续学习的结合；仍主要围绕概念层或类别增量，尚未解决模型机制、模型间信用分配和跨情境组合。 |
| Large Language Models as Tool Makers，ICML 2024 [7] | 否/部分 | 部分 | 部分 | 是 | 是 | 部分 | 是 | 部分 | LLM 可由示例创建可执行工具并在后续检索；工具是程序化能力，不等于带证据边界、可证伪预测和世界机制的认知模型。 |
| CRAFT，ICLR 2024 [8] | 否/部分 | 是 | 部分 | 是 | 是 | 部分 | 是 | 部分 | 从示例生成、检索和组织专业工具集；已经覆盖“创建—验证—检索”中的重要部分，因此 LollaNet 不能把“LLM 生成可复用模块”单独宣称为新颖性。 |
| TRICE，Findings of ACL 2024 [9] | 否 | 是 | 是 | 否 | 否 | 否 | 部分 | 部分 | 通过 execution feedback 学习何时及如何使用工具；直接覆盖结果驱动的工具路由，但没有持久显式模型的语义演化和跨学科机制组合。 |
| ToolGate，ACL 2026 [10] | 否/部分 | 是 | 是 | 部分 | 部分 | 是 | 是 | 是 | 用前置条件、后置条件和验证器约束工具调用及状态提交；它提醒我们“独立验证门”本身并不新，LollaNet 必须提出模型语义残差归因或跨模型信用分配上的新算法。 |
| Gödel Agent，arXiv 2024 [11] | 否 | 是 | 是 | 是 | 是 | 部分 | 是 | 部分 | 通过形式化自反思递归改进 Agent，接近自进化 Agent；修改对象主要是 Agent 程序或策略，不是具有版本化语义契约的显式跨学科模型库。 |
| Darwin Gödel Machine，arXiv 2025 [12] | 否 | 部分 | 是 | 是 | 是 | 部分 | 是 | 部分 | 通过 LLM 生成和基准验证持续修改 Agent 代码，并维护演化档案；它把演化对象放在代码 Agent，LollaNet 需要证明显式机制模型和跨任务路由不是换一种代码变异。 |

### 4.1 自动科学发现方向的补充比较

AI co-scientist、Robin 和 HypoGeniC 说明“LLM 生成假设—批评/排序—实验验证”已经形成独立研究路线。[13–15] 它们的演化对象通常是科学假设、实验方案或研究流程，而不是每次可由路由器调用、具有持久边界和可证伪输出的模型节点。这个区别可能支持 LollaNet 的应用形态，但**不能单独构成算法新颖性**；必须由算法和干预实验证明模型节点的增量作用。

Self-Evolving World Models、POLARIS 等 2025–2026 年工作还表明，环境反馈、世界模型更新和自进化 Agent 的统一训练正在快速发展。[16–17] 因此 LollaNet 必须把“世界转移模型”与“网络自我更新”明确分开，并将语义演化的接受标准写成可复现的统计过程，而不是概念性描述。

## 5. 已经不能单独声称新颖的主张

以下主张各自都有明显先例，不能作为第一篇论文的核心贡献：

- 学习选择或组合模块；
- 学习模块调用顺序和工具使用策略；
- 使用人类可读的概念或中间表示；
- 让 LLM 生成工具、概念、假设或策略；
- 从执行反馈中更新调用策略；
- 用前置条件、后置条件、探针或保留集验证候选；
- 通过失败经验修改 Agent 的程序、提示或策略；
- 把金融时间外回测作为“真实结果”验证层。

这些都可以是 LollaNet 的组成部分，但论文必须避免把它们重新包装成单项创新。

## 6. 当前最可辩护的缺口

在本轮代表性文献范围内，尚未看到一个系统同时满足以下组合：

1. **持久的显式跨学科机制模型节点**：节点不是普通工具或潜在神经模块，而是包含机制、输入输出、成立前提、适用边界、证据和可证伪预测的版本化对象；
2. **结果训练的多步模型网络路由**：网络根据情境和目标学习选择哪些模型、顺序、强度和结构，并对模型间组合进行信用分配；
3. **残差归因后才触发的语义演化**：先用固定语义训练路由，在路由容量和数据问题被排除后，才判定残差可能来自模型语义缺失；
4. **LLM 与接受门解耦**：LLM 只能提出 `amend / add / split` 候选，候选冻结后由封闭回放、保留集和时间外结果决定接受、暂缓或回滚；
5. **跨任务组合分布外证据**：新增或修改模型必须在未参与候选生成的任务和机制组合中仍有增量价值；
6. **干预式忠实性验证**：通过删除节点、打乱语义契约、替换模型或强制路由，证明显式模型确实参与了预测和行动，而不是只改善解释文本。

这只是一个**交叉组合缺口**，尚未自动构成 A 类算法贡献。评审很可能指出：如果 LollaNet 只是把模块路由、概念瓶颈、工具学习和 LLM 反思串在一起，它会被看作系统集成。因此第一篇论文必须把下面的机制形式化为核心算法，而不是把完整架构全部平均铺开。

## 7. 建议冻结的单一核心贡献

### Residual-Gated Semantic Evolution（RGSE，残差门控语义演化）

建议把第一篇论文的主要贡献收敛为一种**双时间尺度、责任可分解的模型演化算法**：

- 高频回路只更新数值参数 `Theta`：路由、连接强度、场景可靠度、置信度校准、顺序和停止条件；
- 低频回路只在“稳定残差”出现时运行：生成并验证语义候选 `Delta Phi`；
- 触发前必须区分三种原因：数据/执行错误、固定模型下的路由不足、模型语义缺失；
- LLM 只生成结构化候选，不参与最终接受；
- 候选在冻结后进行历史回放、反事实消融、封闭保留集和时间外测试；
- 接受标准同时考虑结果提升、校准、复杂度、稳定性和回滚安全。

一个足够清晰的责任分解可以写成：

```text
当前误差
= 数据/执行误差
+ 固定语义下可由路由消除的误差
+ 路由充分后仍持续存在的语义残差
```

设 `L_fixed` 是固定模型语义、经过充分路由优化后的验证损失，`L_current` 是当前系统损失。若误差归因显示 `L_current - L_fixed` 主要来自路由不足，应继续训练 `Theta`；若固定语义下的路由上限仍留下跨 Episode、跨任务重复的残差，才允许生成 `Delta Phi`。候选是否被接受，必须由候选冻结后未见数据上的增量结果决定。

这比“LLM 可以修改模型”更像一个可检验的算法问题：

> **什么时候可以证明，继续调整路由已经不够，应该让语言模型修改显式模型语义？**

## 8. 第一篇论文必须提供的证据

### 8.1 机制可控的合成环境

构造已知生成机制的环境，至少包含趋势、均值回归、反馈增强、瓶颈、延迟、外部冲击、路径依赖和相互作用。训练和测试使用不同机制组合，而不是只改变随机种子。这样可以测量：

- 路由是否选择了真正相关的模型；
- 模型顺序和连接强度是否影响结果；
- 缺失机制出现时，RGSE 能否触发正确的语义候选；
- 候选能否迁移到未见过的机制组合。

### 8.2 强基线与逐步消融

至少比较：

1. 直接神经网络或 Transformer baseline；
2. 固定人工模型组合；
3. 普通 Top-k / MoE 路由；
4. 仅训练 `Theta` 的 LollaNet；
5. 自由 LLM 路由；
6. “LLM 直接修改并立即接受”；
7. 具有独立验证门但没有残差归因的系统；
8. 完整 RGSE。

关键消融不是删掉一个漂亮模块，而是逐一验证因果必要性：

- 去掉残差责任分解；
- 随机或固定时机触发语义演化；
- 取消候选冻结和独立接受门；
- 打乱模型语义契约；
- 删除跨学科模型；
- 固定路由顺序；
- 只允许单步调用；
- 取消结果反馈。

### 8.3 忠实性与防装饰实验

只报告解释文本质量不够。必须进行干预：

- 删除一个模型节点；
- 交换两个模型的语义契约；
- 强制加入一个不适用模型；
- 强制改变调用顺序；
- 将候选模型替换为结构相似但机制相反的模型。

如果模型节点真正在计算中发挥作用，这些干预应使相应预测、置信度或行动发生可预期变化；若性能完全不变，说明显式语义很可能只是提示词装饰。

### 8.4 组合分布外与长期复用

候选模型不能只在触发它的失败集上有效。必须使用：

- 未参与候选生成的任务；
- 未见过的学科组合；
- 不同时间段或机制强度；
- 候选接受后的新 Episode。

报告新增模型的跨任务复用率、增量损失、遗忘、错误迁移和触发成本。金融数据可以作为时间变化的现实验证，但不能作为唯一证据，也不能只报告最高收益；应报告净收益、回撤、尾部损失、换手、校准和多重尝试修正后的指标。

## 9. 初步判断：REVISE，暂不停止

### GO 的部分

- 研究对象清楚：显式模型节点 + 学习路由 + 结果反馈 + LLM 语义候选；
- 具有可操作的实验路径：机制可控环境、路由干预、语义演化和时间外测试；
- 跨学科模型可以提供比普通工具更明确的机制边界和可证伪预测。

### REVISE 的部分

当前 Proposal 把三项贡献写得过于平行：显式模型网络、LLM 语义演化、跨领域评测协议。面对 POLARIS、ToolGate 和 Gödel Agent 等工作，评审可能认为它是已有组件的组合。需要把主贡献改为 RGSE，并把“跨学科模型库”和“金融回测”降为验证场景或实验资产。

Proposal 还需要补上四个形式化定义：

1. 稳定结构性残差的统计触发条件；
2. 路由不足与语义缺失的反事实责任分解；
3. 候选模型的跨任务增量价值与复杂度惩罚；
4. 接受、拒绝、暂缓和回滚的预注册规则。

### STOP 的条件

如果后续实现显示以下任一情况，应停止把 RGSE 声称为算法贡献并改写为系统/基准论文：

- 在固定语义下，普通路由已解释全部收益；
- 语义打乱、节点替换和强制路由几乎不改变结果；
- LLM 候选只在生成集或触发失败集有效，不能跨组合复用；
- POLARIS 或相近强基线在相同预算下达到相同效果；
- 接受门只增加工程复杂度，没有降低错误演化率或回归风险。

## 10. 下一步：冻结研究问题与贡献主张

下一步不应马上扩充更多模型卡，而应先把以下研究问题写成预注册版本：

> **RQ1：在显式模型语义固定时，结果训练的多步路由是否比普通 Top-k 路由更能泛化到未见机制组合？**
>
> **RQ2：当路由优化已达到固定语义下的性能上限时，RGSE 是否能比无门控的 LLM 修改更可靠地发现可复用语义修正？**
>
> **RQ3：候选模型的跨任务增量价值是否来自真实机制作用，而不是解释文本或数据集记忆？**

如果这三个问题不能分别由独立实验回答，论文范围仍然过大。建议先完成：

```text
RQ 冻结
 -> 算法伪代码
 -> 合成机制环境
 -> 强基线实现
 -> 小规模先导实验
 -> 再决定是否进入金融回测
```

## 11. 研究限制与 AI 使用说明

本轮是面向算法定位的定向文献探索，不是 PRISMA 意义上的穷尽式系统综述。检索覆盖的数据库和关键词有限，尤其需要在实验开始前继续检查 2026 年最新会议论文、arXiv 预印本和同一作者的后续版本。矩阵中的“缺口”是基于公开论文描述作出的研究判断，不是优先权或绝对新颖性证明。

本文件由 AI 辅助检索、整理和起草；论文正式提交前必须由研究者逐条打开原始论文，核对作者、版本、实验设置、引用关系和结论边界。

## 参考文献

[1] Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. Neural Module Networks. CVPR, 2016. [arXiv](https://arxiv.org/abs/1511.02799)

[2] Ostapenko, O., Rodriguez, P., Caccia, M., & Charlin, L. Continual Learning via Local Module Composition. NeurIPS, 2021. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html)

[3] Hu, M., Chang, H., Ma, B., Shan, S., & Chen, X. Scalable Modular Network: A Framework for Adaptive Learning via Agreement Routing. ICLR, 2024. [OpenReview](https://openreview.net/forum?id=pEKJl5sflp)

[4] Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. Concept Bottleneck Models. ICML, 2020. [PMLR](https://proceedings.mlr.press/v119/koh20a.html)

[5] Oikarinen, T., Das, S., Nguyen, L. M., & Hu, W. Label-Free Concept Bottleneck Models. CVPR, 2023. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Oikarinen_Label-Free_Concept_Bottleneck_Models_CVPR_2023_paper.html)

[6] Oikarinen, T., et al. Language-Guided Concept Bottleneck Models for Interpretable Continual Learning. CVPR, 2025. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Oikarinen_Language-Guided_Concept_Bottleneck_Models_for_Interpretable_Continual_Learning_CVPR_2025_paper.html)

[7] Cai, T., et al. Large Language Models as Tool Makers. ICML, 2024. [PMLR](https://proceedings.mlr.press/v235/cai24b.html)

[8] Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H., & Ji, H. CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets. ICLR, 2024. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html) / [arXiv](https://arxiv.org/abs/2309.17428)

[9] Qiao, S., Gui, H., Lv, C., Jia, Q., Chen, H., & Zhang, N. Making Language Models Better Tool Learners with Execution Feedback. NAACL, 2024. [ACL Anthology](https://aclanthology.org/2024.naacl-long.195/)

[10] Liu, Y., Peng, X., Cao, J., Wang, X., Deng, S., Chen, J., Yin, J., & Zhang, X. ToolGate: Contract-Grounded and Verified Tool Execution for LLMs. arXiv, 2026. [arXiv](https://arxiv.org/abs/2601.04688)

[11] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement. arXiv, 2024. [arXiv](https://arxiv.org/abs/2410.04444)

[12] Zhang, J., Hu, S., Lu, C., Lange, R., & Clune, J. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv, 2025. [arXiv](https://arxiv.org/abs/2505.22954)

[13] Zhou, Y., Liu, H., Srivastava, T., Mei, H., & Tan, C. Hypothesis Generation with Large Language Models. arXiv, 2024. [arXiv](https://arxiv.org/abs/2404.04326)

[14] Gottweis, J., et al. Accelerating scientific discovery with an AI co-scientist. Nature, 2026. [Nature](https://www.nature.com/articles/s41586-026-10644-y)

[15] Ghareeb, A. E., et al. A multi-agent system for automating scientific discovery. Nature, 2026. [Nature](https://www.nature.com/articles/s41586-026-10652-y)

[16] Self-Evolving World Models: Building Self-Improving Agents with Large Language Models. arXiv, 2025. [arXiv](https://arxiv.org/abs/2511.01171)

[17] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair. arXiv, 2026. [arXiv](https://arxiv.org/abs/2603.23129)

[18] Yao, H., et al. Wild-Time: A Benchmark of In-the-Wild Distribution Shift over Time. NeurIPS, 2022. [NeurIPS](https://papers.nips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html)

[19] Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. The Probability of Backtest Overfitting. Journal of Computational Finance, 2017. [SSRN](https://ssrn.com/abstract=2326253)

[20] Bailey, D. H., & López de Prado, M. The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management, 2014. [SSRN](https://ssrn.com/abstract=2460551)

## 参考文献核验说明

本轮保留的引用均指向可公开核验的论文页面、会议论文集或原始预印本。若某个候选标题的链接解析到无关论文、无法确认标题/作者，或只能由二手页面支持，则不把它写入核心缺口判断。2026 年预印本的版本、作者和最终会议状态应在投稿前再次核对。


