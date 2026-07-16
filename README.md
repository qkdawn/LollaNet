# LollaNet

LollaNet 是一个本地、可审计、可选择性沉淀的认知层：它把用户确认的个人资料、长期目标、准则、思维模型和证据，按需装配到当前的 AI 任务中。

查理·芒格的多元思维模型是重要启发，但不是项目边界。模型也可以来自学科研究、阅读、历史、失败案例、个人观察或新的综合。LollaNet 关心的不是模型数量，而是某个模型是否改变了当前任务的假设、预测或行动。

LollaNet 目前处于 MVP 验证期。第一轮 10 题盲评达到预设的探索性门槛，但这只是单次、同模型族评估；是否能稳定改善回答、迁移到其他模型并带来更好的长期结果，仍需独立复现和 30/90 天复盘。

## 为什么需要多元思维模型

人的收入、声誉、身份或组织指标依赖某个结论时，激励可能偏置证据搜索、解释和建议，形成[激励导致的偏见](01-Models/学科知识/心理学/激励导致的偏见.md)。人也容易把最熟悉的专业、工具或解释框架套到所有问题上，形成“铁锤人倾向”。两者还会相互强化：一个人不仅手里只有一把锤子，激励还可能让他更愿意相信眼前的一切都是钉子。

LollaNet 试图通过以下机制降低这两类风险：

- 对复杂任务执行芒格式双轨分析：先用理性分析轨检查事实、利益、概率和机制，再用心理误判审计轨检查这个判断如何形成；
- 先检查谁从某个结论中获益、谁承担错误后果，把激励结构与证据质量分开分析；
- 从不同学科和机制中只选择 3–7 个真正改变假设、预测或行动的非重复模型，避免由单一框架垄断解释；
- 主动寻找反证、竞争性机制、模型冲突、适用边界和失效条件；
- 不按模型数量投票，而是比较因果机制、证据质量和能够区分不同解释的未来观察；
- 通过可观察预测和延迟复盘，检查当初的判断是否只是受到利益、熟悉感或叙事吸引。

这套方法可以帮助暴露和校正偏见，但不能保证消除偏见。模型选择本身也可能受到激励和“铁锤人倾向”影响，因此来源回引、挑战路线、人工审议和长期复盘仍然不可省略。

## 核心边界

- 当前任务默认留在原来的聊天、仓库或应用中，不会自动进入长期记录。
- AI 只读取会显著改变当前判断的最小私人上下文。
- 复杂任务只保留真正改变假设、预测或行动的非重复模型；不按模型数量投票。
- AI 可以提出模型或个人资料修改建议，但不能静默修改正式认知资产。
- 公共框架不能包含用户画像、真实决策、私人对话、未公开项目或专有资料。

## 当前可用内容

- 88 张带版本和证据边界的公共思维模型卡；
- 一套厂商无关的 [分析协议](04-Prompts/LollaNet%20Analysis%20Protocol.md) 和可直接参考的 [Codex Skill](.agents/skills/lollanet-analyze/SKILL.md)；
- 私人资料、项目、决策、案例和模型提案模板；
- 初始化、创建和校验脚本；
- 10 个 v1 虚构盲测任务、第一轮结果，以及 24 个用于 v2 系统盲测和工程模型消融的虚构任务；
- 一套 [Model Test Protocol v2](07-Docs/model-test-protocol-v2.md)：预注册盲测、模型级消融、真实任务复盘与人工组合复盘的边界。

## 一次分析怎么运行

```text
当前任务
    ↓
建立临时任务模型，确认目标、约束、未知量和风险
    ↓
按需读取最小私人上下文
    ↓
从模型、记忆和挑战三个方向检索可回引材料
    ↓
去重、处理冲突，检查基础概率、证据质量和时间尺度
    ↓
执行芒格式双轨分析：理性分析轨 -> 心理误判审计轨
    ↓
若发现重大判断形成风险，回到受影响的理性步骤复查
    ↓
给出当前判断、成立条件和可验证的下一步
    ↓
用户选择是否保存项目、决策、证据或修改提案
```

简单事实查询、格式转换、机械改写和低风险执行可以直接完成，不必启动多路召回。完整规则见 [LollaNet Analysis Protocol](04-Prompts/LollaNet%20Analysis%20Protocol.md)。

这里的“双轨”是同一分析中的两趟检查，不是两个 Agent 分别给结论，也不按两轨投票。心理误判风险只能触发复查，不能单独证明一个结论错误。

## 快速开始

### 直接试用

如果当前 Codex 环境已安装参考 Skill，可以直接调用：

```text
$lollanet-analyze <需要分析的问题>
```

简单任务会跳过复杂召回。其他 AI 工具可以按照 [LollaNet Analysis Protocol](04-Prompts/LollaNet%20Analysis%20Protocol.md) 实现相同流程；协议不依赖特定厂商、向量数据库或固定模型清单。

### 启用私人认知层（可选）

```bash
git clone https://github.com/qkdawn/LollaNet.git
cd LollaNet
python 06-Scripts/init_private.py
```

初始化脚本只补充缺失文件，不覆盖已有私人内容；`private/` 已加入 `.gitignore`。用 Obsidian 打开仓库根目录，从 [LollaNet Dashboard](00-Start/Dashboard.md) 进入。

私人资料按语义分开保存：

| 文件 | 作用 |
| --- | --- |
| `User Profile.md` | 经历、能力、资源和现实约束 |
| `Value Function.md` | 用户确认的价值、硬约束和内部张力 |
| `Principles.md` | 在具体条件下愿意如何行动 |
| `Long-term Goals.md` | 服务价值、可替换的目标状态 |
| `Current Focus.md` | 当前 6–12 周的资源配置和实验 |

观察记录和复盘是证据，不会自动变成身份、价值、准则或目标。当前焦点是可撤回的短周期配置，不是新的终极价值。

## 选择性保存

真实任务不需要默认创建项目。只有需要长期跟踪时，才保存项目或决策：

```bash
python 06-Scripts/create_project.py <project_id> --name "<项目名称>"
python 06-Scripts/create_decision.py "<决策标题>" --project <project_id>
```

案例入口互斥：

```bash
# 私人经历、未公开项目或内部资料
python 06-Scripts/create_case.py "<案例标题>" --project <project_id>

# 有公开来源的真实案例
python 06-Scripts/create_case.py "<案例标题>" --public-case --source "<来源>"

# 完全虚构的示例
python 06-Scripts/create_case.py "<案例标题>" --public-example
```

- 公开真实案例进入 `09-Cases/<domain>/`，必须有来源、`as_of`、事实与推断边界和时代条件。
- 私人或未公开案例进入 `private/projects/<project_id>/evidence/cases/`。
- 纯虚构示例进入 `08-Examples/cases/`，并标记 `example: true`、`fictional: true`。

公开网页可访问不等于其中的个人数据可以任意收录。完整的数据、隐私、来源和案例约束见 [PROJECT_SPEC](PROJECT_SPEC.md)。

## 模型治理

每张正式模型卡都有稳定 ID、版本和状态：`candidate`、`active`、`contested`、`deprecated`。AI 可以提出 `new`、`amend`、`split`、`merge`、`contest` 或 `deprecate`，但用户必须先审阅提案；接受提案和修改正式模型是两个独立动作。

模型提案使用 `proposed`、`accepted`、`rejected`、`deferred` 状态。创建新模型时，需要引用已经接受的提案：

```bash
python 06-Scripts/create_model.py "模型名称" \
  --source-category 学科知识 \
  --discipline 数学 \
  --subdiscipline 通用 \
  --proposal mp-accepted-proposal-id
```

废弃模型保留历史。模型卡必须说明机制、适用边界、反证、失效条件和可观察预测；详细数据契约见 [PROJECT_SPEC](PROJECT_SPEC.md)。

## 仓库结构

```text
00-Start/       入口、Dashboard 和静态索引
01-Models/      按来源大类和具体领域组织的公共思维模型
02-Checklists/  分析与决策检查表
03-Templates/   模型、项目、案例和私人层模板
04-Prompts/     厂商无关的分析协议
05-Maps/        模型关系图
06-Scripts/     初始化、创建和校验脚本
07-Docs/        评估、计划和历史文档
08-Examples/    纯虚构的公开示例
09-Cases/       有公开可核验来源的真实案例
.agents/        Codex 参考 Skill
private/        本地私人资料，不提交 Git
```

## 验证与当前证据

开发或维护仓库时运行：

```bash
python -m pip install -r requirements.txt
pytest -q
python 06-Scripts/validate_vault.py
```

本地已初始化私人层时，再运行：

```bash
python 06-Scripts/validate_vault.py --include-private
```

第一轮结果为 LollaNet 7 胜、普通回答 1 胜、2 平，事实可靠性没有低于基线。这只支持短期回答质量的初步假设，不证明真实决策已经改善，也不证明结果能迁移到其他模型、评审者或任务分布。完整方法、逐题结果和局限见 [evaluation-results-v1.md](07-Docs/evaluation-results-v1.md)。 新的 [Model Test Protocol v2](07-Docs/model-test-protocol-v2.md) 已提供 24 个虚构任务、工程模型消融设计和记录模板；它尚未产生或声称任何 v2 结果。

## 未来方向

未来重点是把模型库转化为可被不同业务 Skill 按需调用的通用认知能力，而不是让模型直接生成论文、算法或项目报告：

```text
思维模型库 -> 通用认知能力层 -> 算法、研究、项目和产品 Skill
```

七类能力暂定为：**定义**（问题重构）、**解释**（机制建模）、**质疑**（假设与反证）、**求证**（证据与不确定性）、**创造**（方案与迁移）、**决策**（权衡与行动控制）、**学习**（反馈与校准）。它们是可组合接口，不是每次都执行的固定流水线。

### 外部认知输入与模型迭代

这是待验证能力，不代表当前已经实现。拟议链路是：

```text
用户选择的信息源
    -> Horizon 等 RSS 收集层
    -> 内容获取与转换层
    -> 私人暂存区
    -> 模型迭代意见日报
    -> 人工决定：忽略、保留证据、阅读笔记或提出修改提案
```

RSS 条目、文章和案例先是输入材料，不会自动成为正式模型。Horizon 和内容转换工具的接口、授权、部署方式、输出格式和维护状态仍需验证。日报、模型提案和正式模型更新保持独立步骤。详细边界见 [external-input-loop.md](07-Docs/plans/external-input-loop.md)。

相关知识来源可按“来源大类 -> 具体领域 -> 正式模型”组织：时势信号、哲学模型、历史机制、他人失败和学科知识。原始文章、研究和案例只能作为输入材料或证据；只有确认其改变了假设、预测或行动后，才关联已有模型或形成修改提案。完整分类契约见 [PROJECT_SPEC](PROJECT_SPEC.md)。

### 个人资料驱动的模型应用与迭代

这也是待验证能力，不代表当前会自动记录或个性化所有任务。公共模型保持通用机制、证据与边界；在用户明确授权且任务相关时，最小私人上下文可以影响模型选择、参数、风险阈值和行动优先级，但不能改写公共模型事实。用户选择保存的真实调用结果默认进入私人评估或项目复盘，并且只能在人工审议 proposal 后影响私人适配或正式模型版本。详细设计、隐私闸门和分阶段验证见 [personalized-model-evolution-loop.md](07-Docs/plans/personalized-model-evolution-loop.md)。
### 个人资料与行动规划

在用户明确授权后，已确认的个人资料、目标、当前焦点和项目约束可以生成候选待办和番茄钟项目：

```text
个人资料与当前焦点 -> 候选行动 -> 用户确认 -> 待办与番茄钟 -> 时间反馈与复盘
```

它用于落实已经选择的方向，不替用户创造新的价值或永久优先级。任务、日程和专注记录属于私人数据，支持暂停、删除和人工覆盖。

### 审稿式项目与算法研究

未来可以提供从评审者立场审视论文、研究计划和技术项目的 Skill，检查主张、机制、证据、替代解释、基线、可复现性和结论边界；也可以辅助算法研究发现不变量、反例、证明义务和基准实验。它们辅助形成可追溯的审计意见和研究路线，不替代领域专家、正式同行评审、正确性证明或实验结果。

### 多元思维模型与诗歌创作

未来可以探索用多元思维模型辅助诗歌创作。模型不替代诗人的经验、感受和语言选择，而是从意象生成、概念张力、视角转换、结构约束、反向思考和跨学科隐喻等方向提供可组合的创作刺激：

```text
主题、经验或情绪
    -> 选择少量能够产生新张力的异质模型
    -> 生成意象、关系、视角和形式约束
    -> 写作、删改与朗读检验
    -> 人决定作品是否成立
```

诗歌创作不按调用模型的数量评价，也不把模型名称机械堆进诗句。需要验证的是，多元模型是否真正增加了意象的新鲜度、语言的准确性、情感的真实性和作品内部的张力；引用既有作品、风格或材料时，仍须遵守来源和著作权边界。

## 进一步阅读

- [LollaNet Dashboard](00-Start/Dashboard.md)：操作入口、认知资产和常用命令。
- [LollaNet Analysis Protocol](04-Prompts/LollaNet%20Analysis%20Protocol.md)：任何 AI 使用 LollaNet 的现行流程。
- [Model Evolution Protocol](04-Prompts/Model%20Evolution%20Protocol.md)：用户显式触发时，怎样保存、复盘和检验模型调用证据；默认不记录、不自动改模。
- [PROJECT_SPEC](PROJECT_SPEC.md)：产品边界、数据契约、模型字段和隐私规则。
- [Evaluation Protocol](07-Docs/evaluation-protocol.md)：盲评、证据追踪和 30/90 天复盘方法。
- [Public Case Library](09-Cases/README.md)：有公开可核验来源的真实案例。
- `07-Docs/` 中标记为 `superseded` 的文档只保留历史，不作为当前实现依据。
