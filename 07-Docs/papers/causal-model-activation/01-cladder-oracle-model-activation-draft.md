---
type: research_paper_draft
title: LollaNet：基于 CLADDER 的显式因果模型激活与内部表征控制
status: superseded
version: 0.2.1
superseded_by: ../../trainable-cognitive-model-network.md
updated: 2026-07-28
---

# LollaNet：基于 CLADDER 的显式因果模型激活与内部表征控制

> 历史神经控制实验稿，保留作研究记录；不属于当前 LollaNet V4 上下文工程。

**英文暂定题目：** *LollaNet-CMA: Oracle-Guided Activation of Explicit Causal Models on CLADDER*

## 摘要

本文研究一个比完整 LollaNet 更窄、可独立证伪的问题：在冻结基础大语言模型时，将一个具有明确语义身份的因果模型表示为专属 Contrastive Activation Addition（CAA）方向，并依据 CLADDER 的隐藏元数据 oracle 选择正确方向，能否比基础模型、CausalCoT Prompt 和错误内部方向更稳定地改善因果推理。

实验将 CLADDER 的十类查询冻结映射为四个研究专用因果模型族：关联推断、干预识别、个体反事实和中介分解。每个模型族使用只描述计算机制、不包含答案或 gold reasoning 的正负提示对提取一条 CAA 方向。主实验不训练问题到模型的 Router，也不读取现有金融微概念；它直接使用 `query_type` 作为仅对控制器可见的 oracle 标签，以隔离检验“正确模型的内部激活是否有独立价值”。

系统比较 Base、CausalCoT Prompt-only、通用因果方向、正确模型方向、错误模型方向、打乱语义映射、范数匹配随机方向和四方向等权叠加。除最终答案准确率外，实验还报告校准、正确答案概率边际、生成成本、隐藏状态 probe 和模型族间忠实性矩阵。只有正确模型方向在封闭测试和反常识、无语义变量 OOD 测试中稳定优于强基线，并表现出与语义一致的对角干预效应，后续工作才进入“线索 -> 微概念 -> 模型激活”Router 的训练。

本文不声称一个认知模型对应一个神经元，不以类脑比喻证明方法成立，也不把 CLADDER 上的增益外推为通用决策能力。研究目标是建立显式模型语义与内部神经控制之间最小、可审计的因果证据。

## 1. 研究动机

现有 activation steering 表明，修改 Transformer 中间表示可以改变模型输出；RISER 进一步使用轻量 Router 动态选择和组合 reasoning vectors。然而，RISER 的向量库来自正负推理轨迹的聚类，最终 primitive 是由数据几何形成的宽泛潜在方向，不要求每条方向对应一个具有机制、前提、边界和可证伪预测的显式认知模型。

LollaNet 的核心主张更强：显式模型不仅应当被检索或写入 Prompt，还应当真实约束内部计算。但直接实现微概念编码、模型网络、多轮路由、长期可靠度、内部 steering 和联合训练，会产生无法分解的信用分配问题。一旦端到端结果失败，无法判断失败来自：

1. 问题中的线索没有被正确识别；
2. 微概念状态估计错误；
3. 微概念到模型的 Router 错误；
4. 正确模型没有可用的神经控制方向；
5. 控制层位或强度错误；
6. 输出接口没有表达已经存在的内部信号。

因此，第一篇实验先跳过模型发现问题，使用 CLADDER oracle 直接提供正确模型族，只检验最后一个关键接口：

> **当系统已经知道应该激活哪个显式因果模型时，激活该模型对应的内部方向是否带来不可由 Prompt、通用方向或错误方向替代的任务增量？**

若答案是否定的，继续训练微概念 Router 没有信息价值；若答案是肯定的，才有理由研究模型识别、长期可靠度与多模型组合。

## 2. 研究边界

### 2.1 本文研究什么

- 冻结基础 LLM，只读取和干预中间 residual stream；
- 为四个预先定义的因果模型族提取专属 CAA 方向；
- 用 CLADDER `query_type` oracle 选择正确方向；
- 比较正确、错误、随机、打乱和非专属控制；
- 检验控制是否改善答案、校准和内部可分信号；
- 检验收益能否迁移到 anticommonsense 与 nonsense 变体。

### 2.2 本文不研究什么

- 不训练问题到模型的 Router；
- 不训练微概念状态编码器；
- 不把现有金融微概念迁移到 CLADDER；
- 不训练多轮模型网络、模型间连接或停止策略；
- 不进行 HyperSteer、Adapter、LoRA 或基础模型全量微调；
- 不从真实用户历史学习个性化模型可靠度；
- 不修改 `01-Models/`、正式模型状态或公共模型关系；
- 不声称 CLADDER 覆盖现实世界全部因果判断。

### 2.3 与现有 LollaNet 文档的关系

本文是独立研究草稿，不替代：

- [显式认知模型网络与内部神经控制提案](../lollanet-cross-disciplinary-model-network-proposal.md)；
- [可训练认知模型网络架构](../../trainable-cognitive-model-network.md)；
- [Phase 1 Microconcept State Encoder Training Protocol](../lollanet-v1/03-microconcept-state-encoder-training-protocol.md)；
- [RGSE 方法规格](../lollanet-v1/02-rgse-method-spec.md)。

本文只验证上述长期架构中的一个必要条件：显式模型激活到内部控制的接口是否存在可测量价值。

### 2.4 当前实施状态（非实验结果）

截至 2026-07-25，Stage 0 已在 `bundle-v4` 冻结 3,200 条记录，并通过源文件重构、工件哈希、四模型族平衡及跨 split `model_id` 零重叠验证。审计发现，最初的 4B 资格赛进程虽然只生成了 13 条 hook-smoke 记录、尚未生成资格赛预测或 run manifest，但在 split 过滤前曾把包含非资格 split 标签的合并 controller/evaluator 文件载入内存。该进程已停止，工件保留在 `qualification/invalidated/20260724-holdout-exposure/`，并由 invalidation manifest 明确标记为不得用于 selection 或经验性主张。

修复后已分别物化并验证 qualification（200 条）、vector_train（800 条）和 development（400 条）的 split-scoped access view；尚未物化 sealed 或 OOD view，候选进程也不再接受合并 Stage 0 bundle。旧 v0.2.0 候选中的 `Qwen3-4B-Thinking-2507` 虽完成 200 题并通过运行与 hook 硬门，但 200 条输出均达到 512-token 上限，199 条未产生冻结 final-answer 标记，解析失败率为 99.5%，宏平均准确率为 0.5%。该结果只说明 Thinking 候选与冻结生成预算不兼容，不作为模型能力结论；旧候选集及 `Qwen3-8B` 的未完成 hook-smoke 工件不得进入新版 selection。

资格赛章程已升为 v0.3.0，并加入 `reasoning_mode: disabled` 硬约束，Thinking 与 R1-Distill 模型被显式拒绝。新候选为 `Qwen3-4B-Instruct-2507`、`Qwen2.5-3B-Instruct` 和 `Phi-3.5-mini-instruct`；三个 checkpoint 均从可访问的 ModelScope Git-LFS 镜像按独立 commit 冻结，并保存逐文件 SHA-256。当前 readiness 工件为 `D:\LollaNetData\lollanet-cma\qualification\qualification-readiness-v6.json`，内容哈希为 `cd9ead17351f7da7ba649468e5e229334bae44ea50ad08fcb0cbe3d4f5522a3a`，状态为 `ready`；qualification access-view manifest SHA-256 仍为 `57a6054fb60ec57ab322bbc385444800e2be31fde896b69fc93751b0872cd18a`。目前没有候选运行进程，GPU 空闲；Stage 1 方向拟合仍为 `blocked_missing_qualification`，development preflight 仍为 `blocked_missing_directions`。故当前仍没有基础模型胜者、CAA 方向、development 选择、sealed/OOD 结果或 probe 结果。本节只记录实施状态，不构成经验性证据。

sealed/OOD 数据另设独立解封闸门。当前 `D:\LollaNetData\lollanet-cma\evaluation\sealed-release-preflight-v1.json` 的内容哈希为 `f69f4c184c5167bb15c9e1d7fa932c47a1c7c8eaf057acd2ddbb6a5124adb25b`，状态为 `blocked_missing_development`，并明确记录 `unseal_authorized: false` 与 `stage0.validation: null`。只有 development manifest 以 `status: frozen` 通过全部 hard gate 和方向专属性门，且其 18 个条件工件、方向张量、checkpoint、代码与运行环境重新验证后，独立 release 进程才可物化 sealed_test、anticommonsense 和 nonsense 三个隔离 view；后续评估进程不得接收合并 Stage 0 bundle。

无关能力控制已从 MMLU-Pro 的冻结 revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` 构造为独立 200 题 bundle，在 math、physics、chemistry 和 computer science 四个类别中各按固定 seed 与完整源记录哈希确定性选择 50 题。该 revision 的 dataset card 标注许可证为 MIT；原始 test Parquet、dataset card、选择规则、prompt template、记录 ID 和三个隔离工件均保存 SHA-256。prompt 工件不含答案、答案索引、`cot_content` 或源记录哈希，evaluator 标签单独保存。该控制集用于检测非 CLADDER 通用能力负迁移；类别选择不构成“每一道 STEM 题均不涉及因果概念”的内容声明。当前 bundle 内容哈希为 `ef51e923c2010f31454c84874995378fd30470ebb02ec09de1c0e8db32f17c7f`。sealed evaluation preflight 仍为 `blocked_missing_release`，未读取或运行任何 sealed/OOD 题目。

## 3. 研究问题与假设

### 3.1 研究问题

**RQ1：模型激活增量。** 在相同基础模型、输入、解码预算和外部信息下，正确的 oracle 模型专属 CAA 方向是否优于 Base 和 CausalCoT Prompt-only？

**RQ2：语义专属性。** 正确方向是否优于错误模型方向、打乱模型—方向映射、范数匹配随机方向和四方向等权叠加？

**RQ3：分布外稳定性。** 正确方向的收益是否在 anticommonsense 和 nonsense 变体中保持，而不是依赖自然语言中的常识或词汇锚点？

**RQ4：内部信号与表达。** 干预是否增强与正确答案相关的内部可分信号，还是只在输出层附近改变 Yes/No 表达偏置？

**RQ5：模型容量边界。** 模型激活增量是否只存在于某个基础模型，或在本机可运行的不同开放权重模型中具有可复现性？

### 3.2 研究假设

- **H1：** 正确 oracle 模型方向在 sealed test 上的宏平均准确率高于 Base 和 CausalCoT Prompt-only。
- **H2：** 正确方向在对应模型族上的增量高于所有错误、随机和打乱控制，并形成语义对角优势。
- **H3：** 正确方向在 anticommonsense 与 nonsense 测试上的增量不发生方向反转。
- **H4：** 正确方向提高隐藏状态 probe 的答案可分性，且收益不能完全由输出 token 的固定 Yes/No 偏置解释。
- **H5：** 干预不会以不可接受的无关能力损伤换取 CLADDER 增益。

## 4. 研究专用因果模型族

四个模型族是本文的冻结实验语义，不是正式公共模型卡，也不代表因果推断理论只能这样划分。

| 模型 ID | 名称 | CLADDER `query_type` | 核心计算职责 | 主要混淆风险 |
| --- | --- | --- | --- | --- |
| `cm-associational` | 关联推断 | `marginal`, `correlation`, `exp_away` | 在给定联合分布与条件信息下计算边缘、条件关系和解释消除效应，不把观察条件改写为干预 | 将相关关系误作干预效应；忽略 collider conditioning |
| `cm-interventional` | 干预识别 | `ate`, `backadj`, `collider_bias` | 区分观察与 `do(X)`，识别可调整路径、混杂与 collider，并计算或判断总体干预效应 | 用观察条件概率替代干预分布；调整 collider 或遗漏混杂 |
| `cm-counterfactual` | 个体反事实 | `ett`, `det-counterfactual` | 在已知事实条件下执行 abduction、action、prediction，区分总体效应与已接受处理者的反事实 | 把总体 ATE 代替个体条件反事实；混淆事实世界与反事实世界 |
| `cm-mediation` | 中介分解 | `nde`, `nie` | 区分自然直接效应与自然间接效应，保持跨世界变量定义和中介路径一致 | 混淆总效应、直接效应与中介效应；违反识别条件 |

### 4.1 Oracle 可见性边界

`query_type`、`rung`、`graph_id`、`model_id`、`groundtruth` 和数据集 `reasoning` 都属于评估侧元数据。生成模型正常输入只包含 CLADDER 官方推荐的：

```text
background + given_info + question
```

其中：

- `query_type` 只允许 oracle 控制器选择模型方向；
- `answer` 和 `groundtruth` 只允许评分器使用；
- gold `reasoning` 不得进入向量提取提示、生成提示或模型输入；
- `graph_id` 和 `model_id` 只用于分组切分、覆盖审计和泄漏检查；
- 所有条件必须使用相同的正常输入文本。

## 5. 数据与切分

### 5.1 数据来源

主数据使用 CLADDER balanced 数据集及对应 meta-model 文件。运行前必须记录：

```yaml
dataset_name: causalNLP/CLadder
dataset_variant: balanced
upstream_repository: https://github.com/causalNLP/cladder
upstream_commit: 3d2d1169b4b939a09048a6a75956c8972a93cc38
repository_archive_sha256: 1a13ef750a123a2fa214ed4d77774b2f61c77a056d18d99d3a5bf069e01e5380
question_file: cladder-v1-q-balanced.json
meta_model_file: cladder-v1-meta-models.json
license: MIT
question_file_sha256: 7ca1e3c0aa3d0e3ab7f0a31d469c9a7ce738526c7a78b2e5649d17670c27b699
meta_model_file_sha256: f288ed9046a1ef5e39c0dbf6b971b8a3a485887130b7831dc5da4fe2af5d64b5
anticommonsense_file: cladder-v1-q-anticommonsense.json
anticommonsense_file_sha256: 026541452ea3481419f69a4c987fa09ed7985b8f55a3d519254e0ecbccc0759d
nonsense_file: cladder-v1-q-noncommonsense.json
nonsense_file_sha256: 7a22b2a516c1b29106b240a729e24d1f181fa66a3aaaff054fee46e1830a600a
```

若上游版本标识、README 和实际文件名不一致，以冻结 commit、文件路径和 SHA-256 为唯一数据身份，不用“最新版”代替可复现定位。冻结 ZIP 中无语义变量版本的实际文件名是 `cladder-v1-q-noncommonsense.json`，尽管上游 README 将该变体描述为 `q-nonsense`；本文按实际文件名和哈希记录，不静默改名。Hugging Face revision 未参与本次数据冻结，不能用其当前状态替代上述 GitHub commit 资产。

### 5.2 分组切分原则

不得逐题随机切分。同一 `model_id` 生成的题目共享因果结构、参数和背景，跨 split 会造成结构与数值泄漏。切分以 `model_id` 为不可拆分组，并在组级别尽量平衡四个模型族、原始 query type、rung、graph type 和 Yes/No 标签。

主数据切分为：

| Split | 目标题数 | 用途 | 是否可调参 |
| --- | ---: | --- | --- |
| `qualification` | 200 | 本机基础模型资格赛 | 只允许选择基础模型 |
| `vector_train` | 800 | CAA 方向提取和样本过滤 | 允许构造方向，不允许选最终强度 |
| `development` | 400 | 统一层位、统一强度和解析规则 | 允许调参 |
| `sealed_test` | 1,000 | 主结果 | 禁止调参，只运行预注册条件 |

每个 split 目标按四个模型族均衡。若在严格 `model_id` 分组后无法得到精确题数，则采用不放回的最大共同可行样本数，所有模型族使用相同数量，并在章程中记录短缺原因；不得拆分一个 `model_id` 来补齐整数目标。

Stage 0 使用 SciPy HiGHS 混合整数规划分配完整 `model_id` 组。四模型族题数与组互斥是硬约束；query type、graph type、rung 和 Yes/No 分布使用归一化绝对偏差作为软目标。若求解达到预注册时限，可接受经过二进制性、组唯一性和四族精确题数复核的可行 incumbent，但必须保存 `optimal`、目标值、MIP gap、节点数和求解器状态；不得把时限内可行解描述为全局最优平衡。

### 5.3 OOD 数据

从 CLADDER anticommonsense 和 nonsense 变体分别构造 400 题测试集，每个模型族目标 100 题，并满足：

- 与主数据全部 split 的 `model_id` 不重叠；
- anticommonsense 保留因果结构但破坏常识关系；
- nonsense 保留结构和数值但使用无语义变量名；
- 不使用 OOD 结果选择层、强度、模型或解析规则。

冻结源文件中 balanced 与 anticommonsense、无语义变量版本分别存在 1,710 和 1,828 个重叠 `model_id`，因此不得直接抽取整份 OOD 文件。OOD 分配必须在排除全部主 split 已使用 `model_id` 后按完整组求解，并在最终 bundle 中验证任意 split 间交集为零。

### 5.4 无关能力控制集

使用一个冻结的 200 题通用多项选择控制集，题目不得涉及 CLADDER 因果查询。控制集只用于测量 steering 的负迁移，不参与任何参数选择。数据源、许可证、切分和哈希必须在正式运行章程中冻结。

## 6. 本地基础模型资格赛

### 6.1 候选模型

修订后的资格赛冻结以下非 Thinking 开放权重候选：

1. `Qwen/Qwen3-4B-Instruct-2507`
2. `Qwen/Qwen2.5-3B-Instruct`
3. `microsoft/Phi-3.5-mini-instruct`

选择这些候选不是声称它们在所有榜单上最强，而是因为它们同时接近以下本地实验约束：开放权重、纯 Instruct 答案模式、可由 Transformers 加载、可以读写中间隐藏状态，并可能通过 8GB VRAM 上的 4-bit 量化运行。资格赛章程要求 `reasoning_mode: disabled`，模型 ID 含 `Thinking`、`DeepSeek-R1` 或 `R1-Distill` 时运行器直接拒绝。

### 6.2 统一加载条件

资格赛使用相同条件：

```yaml
quantization: bitsandbytes NF4 4-bit
compute_dtype: bfloat16 when supported, otherwise float16
batch_size: 1
maximum_input_tokens: 2048
maximum_new_tokens: 512
sampling: greedy
temperature: 0
seed: fixed and recorded
reasoning_mode: disabled
```

模型必须通过以下硬门：

1. 在 RTX 4070 Laptop 8GB VRAM 上完成加载；
2. 能通过 forward hook 读取目标层 residual stream；
3. 能修改 residual stream 并继续完成生成；
4. 运行 50 题 hook smoke test 无 OOM、NaN 或输出解析崩溃；
5. 量化前后 tokenizer、chat template 和 Yes/No 解析已冻结；
6. 峰值显存和中位延迟可被稳定记录。

在通过硬门的模型中，以 200 题 `qualification` 集的四模型族宏平均准确率选择唯一基础模型。若差异落在配对 bootstrap 置信区间内，依次使用以下 tie-break：

1. 峰值显存更低；
2. 每题中位延迟更低；
3. 输出解析失败率更低。

基础模型一旦选定，必须在构造 `sealed_test` 运行清单前冻结模型 ID、revision、量化配置、chat template 和依赖版本。

## 7. 模型专属 CAA 方向

### 7.1 正负提示对

对每个模型族定义一对长度和语气尽量匹配的机制指令：

- 正提示要求使用该模型族的合法变量、识别条件和计算步骤；
- 负提示要求使用另一个明确不适用的模型族，或要求忽略该模型族的关键区别；
- 两者都不得包含当前题目的答案、数值结论或 gold reasoning；
- 正负条件使用完全相同的题目集合；完成正确性与 margin 过滤后，在每个模型族内按 gold Yes/No 等量保留样本，并以冻结种子和记录 ID 哈希确定性裁剪多数标签，防止方向混入答案先验；
- 正负提示模板在运行前冻结，并作为附录完整公开。

例如，干预识别的正提示强调区分 `P(Y|X)` 与 `P(Y|do(X))`、检查 backdoor 和 collider；其负提示使用纯观察关联计算，但不能直接暗示最终 Yes/No。

### 7.2 样本过滤

在 `vector_train` 上分别运行正负提示。一个样本对只有在以下条件全部满足时才进入方向估计：

1. 正条件输出可解析；
2. 正条件答案正确；
3. 正条件的正确答案 log-prob margin 高于负条件；
4. 正负提示都没有包含标签泄漏字段；
5. 当前 `model_id` 只属于 `vector_train`。

若某个模型族过滤并完成 Yes/No 等量化后少于 100 个有效样本，该方向标记为 `insufficient`，主实验停止，不使用其他 split 补齐。等量化只允许删除多数标签样本，不能复制少数标签样本。

其中 log-prob margin 使用冻结的规范答案串 `Final answer: yes` 与 `Final answer: no` 计算：对每个答案串取逐 token 平均对数似然，gold 串减去另一标签串得到 margin。正负条件使用同一评分方法；不从自由生成文本中挑选 token，也不因推理长度改变 margin 定义。方向激活取自该 teacher-forced 评分 forward 中答案串之前的最后一个输入 token，避免把 gold 标签 token 本身写入方向。

### 7.3 方向计算

设基础模型第 `l` 层在读取完整输入提示后的最后一个 token 激活为 `h`。对模型族 `i`：

$$
d_{ij}=h^{+}_{ij,l}-h^{-}_{ij,l}
$$

$$
v_{i,l}
=
\operatorname{Normalize}
\left(
\frac{1}{N_i}\sum_{j=1}^{N_i}d_{ij}
\right)
$$

同时计算一个通用因果方向：

$$
v_{general,l}
=
\operatorname{Normalize}
\left(
\frac{1}{\sum_i N_i}
\sum_{i=1}^{4}\sum_{j=1}^{N_i}d_{ij}
\right)
$$

该方向直接汇总所有入选 record-level contrasts，不先把每个模型族均值归一化。因而它与 `equal_all` 不同：后者对四条已经单位化的模型族方向等权求和再归一化。若把两者都定义成四条单位方向的归一化均值，它们会成为完全相同的条件，无法检验“通用 pooled 因果激活”与“同时激活全部专属模型”的差异。方向拟合工件必须同时保存 `general__layer_<l>` 与 `equal_all__layer_<l>` tensor、各自 SHA-256 和构造元数据；sealed preflight 重新由四条 family tensor 计算 `equal_all`，并要求其与保存值一致，且与 `general` 的余弦相似度不高于 0.999999，否则评估不获授权。

每条方向保存：基础模型 revision、层位、hidden size、样本 ID、提示版本、向量范数、SHA-256、dtype 和提取代码版本。不同基础模型的方向不可互换。

### 7.4 层位与强度

只在 `development` 上选择一个所有模型族共享的干预层。候选层为模型深度的约 25%、50% 和 75% 位置，取实际存在的最近整数层。选择标准是四模型族正确方向的宏平均准确率增量，同时要求随机方向和错误方向没有同等增益。

强度使用统一候选集：

$$
\alpha\in\{0.25,0.5,1.0,1.5,2.0\}
$$

只选择一个所有模型族共享的 `alpha`。不得为每个 query type、模型族或 OOD 变体单独调参。

### 7.5 注入方式

注入与 RISER 的 sequence-level controller 保持可比：

$$
h'_{l,t}=h_{l,t}+\alpha v_i
$$

- 首次 forward 在选定层对最后一个输入 token 注入；
- 自回归生成时，在每个解码步对当前最后 token 持续注入；
- 非最后 token 不注入；
- 所有对照使用相同生成预算和 KV cache 设置；
- 向量 dtype 与 residual stream 一致；
- 任何 clipping 或 norm rescaling 必须在 development 前预注册。

## 8. 对照条件

所有条件使用同一个冻结基础模型、数据顺序、正常输入、生成长度、解码参数和答案解析器。

| 条件 ID | 条件 | 目的 |
| --- | --- | --- |
| `base` | 无额外提示、无内部控制 | 基础能力 |
| `causal_cot` | CLADDER CausalCoT Prompt-only，无内部控制 | 强文本推理基线 |
| `general_caa` | 注入通用因果方向 | 检验专属方向是否优于一般因果 priming |
| `oracle_correct` | 由 `query_type` 选择正确模型族方向 | 主方法 |
| `oracle_wrong` | 选择预注册的机制不兼容方向 | 检验语义方向性 |
| `shuffled_mapping` | 固定随机置换模型族—向量对应关系 | 检验名称或数量效应 |
| `random_norm_matched` | 注入与正确方向同范数的随机方向 | 检验一般残差扰动 |
| `equal_all` | 四条方向等权求和后归一化 | 检验激活全部模型是否足够 |

`oracle_wrong` 使用固定循环映射：

```text
associational -> mediation
interventional -> associational
counterfactual -> interventional
mediation -> counterfactual
```

`shuffled_mapping` 在实验章程中由固定种子生成一次，之后不得根据结果更换。

为生成完整的语义忠实性矩阵，实际模型运行不是只执行三种 oracle 路由。四条模型族方向分别在每个测试 split 的全部题目上运行一次，形成 `family_direction__<family>` 四个基础条件。`oracle_correct`、`oracle_wrong` 和 `shuffled_mapping` 再按每条记录的 controller 家族，从这四组已冻结原始输出中确定性选择对应行；派生过程保存源 condition、源行哈希和映射版本，不重新调用模型。由此既覆盖全部 (4\times4) 方向—真实家族组合，也避免把完全相同的方向—题目生成重复计算三次。

`causal_cot` 以 CLADDER 冻结 commit `3d2d1169b4b939a09048a6a75956c8972a93cc38` 中 `causalbenchmark/eval/run_gpt_v1.5_gpt4_1106_cot.py` 的五步 guidance 为来源（Git blob `cee158e498832c4c9af508c332e4c560536bbb48`，源文件 SHA-256 `89e45ac90874a3cf1f4a20c78eb066be380a81c842d9a78b7d3a0a1d7d6b9d5a`）。本研究使用单轮适配版：保留“图提取—query type 判断—estimand 形式化—数据收集—求解”顺序，但不注入上游脚本使用的 `query_type`、变量符号映射或 gold reasoning 元数据，并统一使用本文冻结的 final-answer 协议。完整适配文本及其哈希随实验章程保存。

## 9. 输出解析与指标

### 9.1 答案协议

每个条件都要求模型在响应末尾输出一个可解析的 `yes` 或 `no`。解析器：

1. 优先读取预注册 final-answer 标记；
2. 否则读取响应末尾最后一个独立 `yes/no`；
3. 同时出现冲突答案或没有答案时记为解析失败；
4. 解析失败计为错误，不进行人工修复。

### 9.2 主指标

- 总体准确率；
- 四个模型族的宏平均准确率；
- 每个原始 `query_type` 的准确率；
- Pearl causal ladder 每个 `rung` 的准确率；
- 解析失败率。

### 9.3 次指标

- 正确答案 token 的 log-prob margin；
- Brier score；
- Expected Calibration Error；
- 输出 token 数；
- 每题端到端延迟；
- 峰值 GPU 显存；
- steered 与 base 输出分布的 token-level KL；
- 无关控制集准确率变化。

Token-level KL 只用于内部激活干预条件。对干预条件自身 greedy 轨迹的每一个解码决策，顺序执行两个 batch-size 1 forward 并维持两条独立且条件一致的 KV cache：两侧具有完全相同的 prompt 和已生成 token 前缀，一侧在每一步开启 steering，另一侧始终关闭 steering；每一步由 steered 侧 greedy 选出的 token 同时作为两条轨迹的下一步输入。由此分别得到干预分布与在相同文本轨迹上端到端无 steering 的参考分布，并报告

$$
\frac{1}{T}\sum_{t=1}^{T}D_{KL}\left(P_{\mathrm{steered}}(\cdot\mid x,y_{<t})\parallel P_{\mathrm{unsteered}}(\cdot\mid x,y_{<t})\right).
$$

两侧 logits 以 float64 `log_softmax` 计算；若轨迹生成 EOS，则 EOS 决策计入平均。该定义不强行对齐已经分叉的 steered 与 Base 文本轨迹。Prompt-only CausalCoT 不报告该激活 KL；三个路由派生条件从其逐题 family-direction 源行继承哈希链接的 KL 值。

### 9.4 语义忠实性矩阵

对每条模型方向 `v_i`，在每个真实模型族 `j` 上计算相对 Base 的准确率增量：

$$
F_{ij}=Acc(v_i\mid family_j)-Acc(base\mid family_j)
$$

若方向具有语义专属性，应观察到：

$$
F_{ii} > \frac{1}{3}\sum_{j\neq i}F_{ij}
$$

本文不要求所有非对角项为负，因为模型机制可能共享计算步骤；但至少三个模型族必须表现出预注册的显著对角优势。若所有方向对所有任务产生相近增益，更支持“通用计算激活”而不是“显式模型激活”。

### 9.5 隐藏状态 probe

为区分内部信号增强与表面答案偏置，在冻结层位上训练一个简单的 L2 正则线性 probe，预测正确 Yes/No：

- probe 只使用 `vector_train` 训练；
- 正则强度只使用 `development` 选择；
- 在 `sealed_test` 上分别读取 Base、正确方向、错误方向和随机方向的隐藏状态；
- 报告 probe accuracy、AUROC 和 margin；
- probe 结果只作为机制证据，不能替代生成答案准确率。

Probe 的目标为 gold Yes/No 标签，输入为 development 所选层在正常 prompt 最后一个非 padding token 处的 residual output；steered 条件读取该层完成干预后的激活。训练和正则选择分别只使用 Base 条件的 `vector_train` 与 `development` 激活，禁止使用生成文本或 gold answer token。特征均值和总体标准差只由 `vector_train` 估计，标准差下限为 $10^{-6}$。L2 网格冻结为 $\{10^{-4},10^{-3},10^{-2},10^{-1},1,10,100\}$，依次按 development accuracy、AUROC、gold margin 和更大 L2 打破并列；选定后才在 sealed test 的 Base、正确、错误和注册主随机方向上评估。

若输出准确率改善但 probe 信号不变，并且改善集中在输出层或固定答案 token，结论应收缩为“输出接口控制”，不能声称因果推理内部表示得到增强。

## 10. 统计分析

- Base、CausalCoT 和确定性方向条件在 greedy、temperature 0 与冻结运行环境下各运行一次；重复更换 seed 不构成独立随机重复，不把完全相同的确定性输出当作额外样本；
- 范数匹配随机方向使用一个预注册主 draw 和两个独立 seed 的敏感性 draw；注册比较使用主 draw，另外两个只检验结论是否由单一随机扰动驱动；
- 样本顺序、随机方向和 shuffle mapping 使用彼此独立的预注册种子；
- 条件比较使用题目级配对 bootstrap，至少 10,000 次重采样；
- 报告点估计、95% 置信区间和原始题目级预测；
- 同时报告绝对准确率和相对 Base 增量；
- 主要假设族预先限定为 H1–H5，不从大量未注册切片中选择显著结果；
- query type 和 graph type 切片作为诊断，不替代主指标；
- 不删除失利模型族、解析失败或 OOD 退化结果。

五个注册比较固定为 `oracle_correct` 分别对 Base、CausalCoT、`oracle_wrong`、`shuffled_mapping` 和主随机 draw；只有模型族分层的配对宏平均准确率差 95% bootstrap CI 下界高于 0 才计为通过。每条 family direction 的对角优势也在四个真实 family 内分别重采样 10,000 次；四个 family 按冻结顺序使用 `bootstrap_seed + 100 + family_index`，至少三个 family 的 95% CI 下界必须高于 0。两个额外随机 draw 只要求正确方向的宏平均准确率点估计更高，不作为新的注册显著性检验。

“不由单一 query type 驱动”冻结为：逐一删去每个 query type 后，`oracle_correct - Base` 的四 family 宏平均准确率差仍均大于 0。“不由 parser 异常驱动”冻结为：只在两条件均解析成功的配对题目上宏平均差仍大于 0，且正确方向的总体解析失败率相对 Base 增幅不超过 1 个百分点。无关控制门取四条 family direction 相对 Base 的最差准确率变化，下降不得超过 2 个百分点；边界判断只使用 $10^{-12}$ 浮点容差。

## 11. 成功门与停止规则

### 11.1 进入第二阶段的必要条件

以下条件必须同时满足：

1. `oracle_correct` 在 `sealed_test` 上显著优于 `base`；
2. `oracle_correct` 显著优于 `causal_cot`；
3. `oracle_correct` 显著优于 `oracle_wrong`、`shuffled_mapping` 和 `random_norm_matched`；
4. 至少三个模型族具有显著语义对角优势；
5. anticommonsense 与 nonsense 上的宏平均增量均不发生方向反转；
6. 四条 family direction 分别运行独立无关控制集，其中最差准确率相对 Base 的下降不超过 2 个百分点；
7. 增益不由单一随机方向 draw、单一 query type 或解析器异常驱动，且两个额外随机方向敏感性 draw 不改变注册比较的方向；
8. 每条方向有足够的有效提取样本且来源完全位于 `vector_train`。

### 11.2 失败后的结论收缩

- 若 CausalCoT 与正确方向等效：停止声称内部控制必要，转为 Prompt-only 模型应用研究；
- 若通用方向与专属方向等效：停止声称显式模型具有独立神经载体，转为通用因果 priming；
- 若错误、随机或打乱方向同样有效：停止语义忠实性主张，检查范数、层位和输出偏置；
- 若只有自然语言版本有效、nonsense 失效：结论收缩为词汇或常识增强；
- 若 probe 改善但输出不改善：研究问题转为内部信号到语言表达的接口失败；
- 若输出改善但 probe 不改善：研究问题转为输出控制而非内部推理；
- 若基础模型在资格集接近随机且所有 steering 无效：先更换本地可行基础模型，不训练 Router；
- 若无关能力下降超过门槛：停止增加强度或联合训练，优先研究局部 token、局部层位或稀疏控制。

任一核心门失败，都不得通过增加更多微概念、更多 Router 层或联合训练掩盖失败。

## 12. 第二阶段：微概念中介的模型识别

第二阶段不是本文主实验的组成部分，只有第 11 节全部通过后才启动。

### 12.1 研究专用因果微概念

候选微概念仅作为未来数据契约草案：

- `query_level_signal`：关联、干预或反事实请求；
- `treatment_outcome_roles`：处理变量与结果变量是否可识别；
- `conditioning_structure`：给定条件和目标条件的角色；
- `confounder_path`：共同原因与 backdoor 路径；
- `collider_path`：collider 及其条件化状态；
- `mediator_path`：中介节点与直接、间接路径；
- `instrument_signal`：工具变量及排除限制；
- `intervention_operator`：是否需要 `do(X)`；
- `factual_conditioning`：已观察事实对个体反事实的约束；
- `identification_status`：目标量是否可由给定分布识别。

这些概念不得直接写入公共 `research/phase1/microconcepts/`。第二阶段先在本论文目录的实验资产中定义、标注和消融，验证成功后再提出正式模型或微概念提案。

### 12.2 四种系统比较

```text
Direct Vector Router:
question -> vector weights

Direct Model Router:
question -> model activation -> vectors

Microconcept-Mediated Router:
question -> microconcept states -> model activation -> vectors

Oracle Model Activation:
query_type oracle -> model activation -> vectors
```

Oracle 是性能上限与归因参照，不是可部署方案。第二阶段的主要问题是：微概念中介是否在未见故事、无语义变量和未见图结构上缩小 learned Router 与 oracle 的差距，并且打乱微概念后性能下降。

### 12.3 与现有 Router 的可复用边界

未来可以复用现有 [model activation contract](../../../research/phase1/microconcept_router_v1/contracts/model_activation_v1_contract.yaml) 的独立 sigmoid 思想，使多个模型可以同时相关。但不得复用：

- 金融微概念定义；
- 现有固定 11 节点 ABI；
- Weak 双教师模型有用度分数作为真实可靠度；
- 现有金融 `microconcept -> model` 合法边；
- 单一 `all_activation` 数值同时表示适用性、可靠度和控制增益。

后续系统必须区分：当前证据匹配、任务相关性、情境可靠度和实际 steering gain。

## 13. 有效性威胁

### 13.1 构念效度

- CLADDER 正确答案不等于现实因果决策能力；
- 一个模型族方向可能只编码提示格式或答案倾向；
- query type 的四族映射是工程划分，不是因果理论的唯一分类；
- CAA 的线性方向可能不足以表示多步或非线性推理。

### 13.2 内部效度

- 正负提示长度、措辞和答案分布不匹配会制造伪方向；
- 根据正确答案过滤向量样本可能引入选择偏差；
- 层位与强度搜索可能在 development 上过拟合；
- 量化可能改变激活几何；
- 持续注入可能主要改变输出分布而非推理过程。

### 13.3 外部效度

- 单一基础模型的方向不可迁移到其他模型；
- 形式化、封闭世界的 CLADDER 不能代表开放世界因果识别；
- nonsense 稳定性不自动证明现实语言鲁棒性；
- 四个因果模型族不能支撑 LollaNet 全部跨学科模型的有效性。

### 13.4 结论效度

- 多个条件和切片会增加偶然显著结果；
- 小模型的容量上限可能掩盖有效 steering；
- 输出准确率可能低估已经编码但无法表达的因果信号；
- probe 可分性不证明 probe 读取的表示参与最终生成。

## 14. 复现与审计产物

正式实验必须保存以下只读产物：

```text
experiment_charter.yaml
dataset_manifest.json
split_manifest.json
base_model_qualification.json
prompt_templates/
vector_manifest.json
vectors/*.safetensors
development_selection.json
sealed_release_preflight.json
evaluation_release_manifest.json
sealed_run_manifest.json
predictions/<condition>/<seed>.jsonl
faithfulness_matrix.json
probe_results.json
statistical_report.json
failure_report.md
```

每条预测至少记录：题目 ID、split、模型族、query type、rung、条件、seed、基础模型 revision、层位、alpha、向量哈希、原始输出、解析答案、正确性、log-prob、token 数、延迟和峰值显存。

训练、开发和 sealed 产物必须具有独立 manifest；sealed 结果生成后不得回写层位、强度、提示模板、解析规则或样本过滤规则。

## 15. 预期贡献与主张边界

若成功，本文最多支持以下主张：

1. 一组预先定义的显式因果模型可以对应具有任务增量的内部控制方向；
2. oracle 选择正确模型方向优于通用、错误和随机方向；
3. 语义到神经控制的接口值得继续训练模型识别 Router；
4. CLADDER 可作为显式模型激活的机制可控验证环境。

本文不支持以下主张：

- LollaNet 完整模型网络已经成立；
- 微概念中介已经优于直接 Router；
- steering 方向等同于人类直觉或脑区机制；
- CLADDER 上的因果能力自动迁移到现实决策；
- 一条固定线性方向足以表示所有认知模型；
- 系统已经能够从真实结果长期演化模型语义。

## 16. 研究推进顺序

```text
Stage 0: 冻结数据、四模型族、提示与统计章程
    -> 运行三模型本地资格赛
    -> 冻结唯一基础模型

Stage 1: 提取四条模型专属 CAA 方向
    -> 过滤有效正负样本
    -> 在 development 选择统一层位和 alpha
    -> 冻结向量与运行 manifest

Stage 2: 运行 sealed 主实验与 OOD 实验
    -> 生成忠实性矩阵和 probe 结果
    -> 执行成功门与停止规则

Stage 3: 仅在全部核心门通过后
    -> 定义研究专用因果微概念
    -> 训练 microconcept -> model Router
    -> 比较 direct、mediated 和 oracle routing

Stage 4: 仅在微概念中介具有独立价值后
    -> 研究模型组合、长期可靠度与结果信用分配
    -> 决定是否形成公共模型或架构 proposal
```

## 17. 一句话定义

> **本文用 CLADDER oracle 隔离检验：当正确的显式因果模型已知时，把该模型映射为专属内部激活方向，是否能以可审计、具有语义专属性的方式改善冻结 LLM 的因果推理。**

## 参考文献

1. Jin, Z., et al. *CLadder: Assessing Causal Reasoning in Language Models*. NeurIPS, 2023. [arXiv](https://arxiv.org/abs/2312.04350) · [Code and data](https://github.com/causalNLP/cladder)
2. Pearl, J., & Mackenzie, D. *The Book of Why: The New Science of Cause and Effect*. Basic Books, 2018.
3. Turner, A. M., et al. *Activation Addition: Steering Language Models Without Optimization*. 2023. [arXiv](https://arxiv.org/abs/2308.10248)
4. Rimsky, N., et al. *Steering Llama 2 via Contrastive Activation Addition*. ACL, 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.828/)
5. Ye, W., et al. *RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation Steering*. Findings of ACL, 2026. [ACL Anthology](https://aclanthology.org/2026.findings-acl.226/) · [arXiv](https://arxiv.org/abs/2601.09269)
6. Sun, J., et al. *HyperSteer: Activation Steering at Scale with Hypernetworks*. 2025. [arXiv](https://arxiv.org/abs/2506.03292)
7. Ding, Z., & Zhang, X.-P. *Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express*. 2026. [arXiv](https://arxiv.org/abs/2605.25891)
8. Koh, P. W., et al. *Concept Bottleneck Models*. ICML, 2020. [PMLR](https://proceedings.mlr.press/v119/koh20a.html)
