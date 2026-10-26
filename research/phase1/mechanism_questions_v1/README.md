# LollaNet 跨领域机制问题集 V1

本目录实现 160 个公开事实包、640 个机制问题 Episode 及双轮 Weak 标签的构建、隔离和验证。正式的 11 个微概念、20 个思维模型及其固定关系只读；生成器不会为了覆盖配额修改语义或伪造负例。

## 完整数据流

```text
官方结构化来源 + 本地 source-cache
  -> 160 个事实包（四领域各 40）
  -> 事实证据微概念状态（case-level baseline）
  -> 每包四种任务问题
  -> 题型需求与事实微状态共同生成 fact-specific task_relevance 向量
  -> 控制反事实的 status/intensity/direction/temporal_persistence/task_relevance 声明实际写入 Episode
  -> route-blind external teacher packet
  -> 外部 LLM initial Weak route
  -> 不同外部 LLM 的 independent contract audit
  -> accepted / review_required
```

真正的 Router 输入仅包含用于关联的 `episode_id`、`microconcept_states` 的五个轴和固定索引导出的 `candidate_subgraph`；它不读取 `question`、`goal`、`constraints`、事实、来源或案例身份。问题语义通过显式的 `task_relevance_profile` 和 `task_focus_microconcept_id` 与事实包中的状态显著度共同写入 `task_relevance` 向量；Episode 额外保留 Router 不可见的 `task_relevance_basis`，逐概念记录 profile 水平、事实水平、focus bonus 和最终值，避免同一题型退化成近固定模板。教师 packet 是独立的教师可见视图：它保留 `question`、`goal`、`constraints` 以便外部教师理解任务，但仍不包含事实、来源或案例身份；教师可见不等于 Router 可见。

## V1 官方来源

当前生产器固定使用四个可缓存、可复核的官方结构化来源：

- `business_finance`：SEC XBRL Frames；
- `software_engineering`：CISA Known Exploited Vulnerabilities Catalog；
- `product_organization`：U.S. CPSC Public Recall REST API；
- `science_public_systems`：USGS FDSN Earthquake API。

原始响应仅缓存在外部运行时目录的 `source-cache/`；仓库中的公开案例卡只保存必要摘要、时间截点和来源定位，不复制原始全文。当前四个来源族保证了结构化与可复核性，但并不等于已经覆盖每个领域的全部现实机制；覆盖不足的模型会继续标记为 `unsupported`。

## 构建与外部教师流程

构建阶段只产生 Episode 与待标注的 route-blind packet，**绝不**产生哈希、规则或伪 LLM 标签：

```powershell
python -m research.phase1.mechanism_questions_v1.build_public_corpus_v1 `
  --output D:\LollaNetData\mechanism-question-set-v1.1 `
  --source-runtime-root D:\LollaNetData\mechanism-question-source-runtime-v1.1

# external initial LLM (train/development only)
python -m research.phase1.mechanism_questions_v1.llm_teacher_v1 `
  --mode initial `
  --packets D:\LollaNetData\mechanism-question-set-v1.1\teacher_packets\train_development\initial.jsonl `
  --episodes D:\LollaNetData\mechanism-question-set-v1.1\episodes\train_development.jsonl `
  --model <initial-model> --teacher-version <pinned-version> `
  --output D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\initial.jsonl `
  --audit-ledger D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\initial-audit.jsonl

python -m research.phase1.mechanism_questions_v1.prepare_gpt_teacher_packets_v1 `
  --mode review `
  --episodes D:\LollaNetData\mechanism-question-set-v1.1\episodes\train_development.jsonl `
  --initial-labels D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\initial.jsonl `
  --output D:\LollaNetData\mechanism-question-set-v1.1\teacher_packets\train_development\review.jsonl

# must be a different model identifier from <initial-model>
python -m research.phase1.mechanism_questions_v1.llm_teacher_v1 `
  --mode review `
  --packets D:\LollaNetData\mechanism-question-set-v1.1\teacher_packets\train_development\review.jsonl `
  --episodes D:\LollaNetData\mechanism-question-set-v1.1\episodes\train_development.jsonl `
  --model <independent-audit-model> --teacher-version <pinned-version> `
  --output D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\review.jsonl `
  --audit-ledger D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\review-audit.jsonl

python -m research.phase1.mechanism_questions_v1.finalize_teacher_labels_v1 `
  --episodes D:\LollaNetData\mechanism-question-set-v1.1\episodes\train_development.jsonl `
  --initial-labels D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\initial.jsonl `
  --review-labels D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development\review.jsonl `
  --output-dir D:\LollaNetData\mechanism-question-set-v1.1\labels\train_development
```

原始官方响应的 `source-cache/` 必须位于训练数据根目录之外，通过 `--source-runtime-root` 指定；这样训练进程只拿到开放 Episode 与标签，不能从完整原始缓存重建 sealed case。

默认开放根目录只写 `train` 与 `development`：`fact_packs.jsonl`、`source_ledger.jsonl`、Episode 和 teacher packet 都不会包含 `sealed_test`。sealed materialization 必须同时给出 `--include-sealed --confirm-sealed` 两个显式确认；产物默认进入 `<output>\sealed\`，也可用 `--sealed-output` 指向另一个独立目录，绝不与开放根目录混写。构建器仍不会自动标注或评估 sealed 数据。若 prepare 工具选中的 Episode 含 `sealed_test` 而双重确认不完整，它会 fail-closed，不会静默写包。

## 正式验收

```powershell
# This input is the access-controlled, independently curated full 160-pack source,
# not the filtered fact_packs.jsonl written by the default open build.
python -m research.phase1.mechanism_questions_v1.build_question_episodes_v1 `
  --fact-packs D:\LollaNetData\mechanism-question-source-v1\fact_packs.full-input.jsonl `
  --output D:\LollaNetData\mechanism-question-set-v1\build `
  --require-v1-size

# --require-v1-size validates the complete source before partitioning;
# build\episodes.jsonl remains train/development-only. Do not feed the filtered
# open fact_packs.jsonl back into the splitter because it no longer contains all case IDs.
python -m research.phase1.mechanism_questions_v1.validate_mechanism_questions_v1 `
  --episodes D:\LollaNetData\mechanism-question-set-v1\build\episodes.jsonl `
  --labels D:\LollaNetData\mechanism-question-set-v1\labels\train_development\all_labels.jsonl `
  --report D:\LollaNetData\mechanism-question-set-v1\reports\train_development_validation.json

python 06-Scripts/validate_vault.py
python -m pytest -q
```

完整 V1 额外要求至少 160 种不同题面文本，防止用四个完全相同的模板伪装成 640 个问题。

## 受控反事实

每个反事实 `change` 必须：

- 指向已存在的 `fact_id`；
- 修改该事实对象上真实存在的字段；
- `before` 与原事实字段值完全一致；
- 只改变 1–2 个事实字段；
- `microconcept_state_changes` 可声明 `status`、`intensity`、`direction`、`temporal_persistence`、`task_relevance`；
- 至少声明一个非 `task_relevance` 状态轴，且每个 `before` 必须匹配问题条件化后的状态；Episode 构建后逐轴核对 `after` 已真正生效。

构建器会在开放输出的 `reports/question_conditioning_audit.json` 写入审计结果，报告每个 `question_kind` 的唯一 relevance 向量数、主导模板占比、同题型异事实向量差异率，以及反事实各状态轴的声明/应用数量。也可独立执行：

```powershell
python -m research.phase1.mechanism_questions_v1.audit_question_conditioning_v1 `
  --episodes D:\LollaNetData\mechanism-question-set-v1.1\episodes\train_development.jsonl `
  --report D:\LollaNetData\mechanism-question-set-v1.1\reports\question_conditioning_audit.json
```

## 外部教师边界

`weak_teacher_v1.py` 的历史哈希式选择器已禁用，不能回退使用。`llm_teacher_v1.py` 需要配置 `OPENAI_API_KEY`，并将外部 initial 模型与 audit 模型的标识符记录在每条标签中；若两者相同，审计阶段会失败。每次调用另写非敏感 audit ledger，其中包括 packet、响应的 SHA-256 指纹、模型、版本和 API response id，而不保存密钥或原始响应。

本机仍对每个初标与审计标签执行选择契约验证。只有 `review_status: accepted` 的审计标签可进入 Router；它们依旧是 Weak supervision，不是 Gold、事实真值或模型语义。

## 隔离布局

开放根目录与 sealed 根目录物理分开：

```text
<output>/
  fact_packs.jsonl                 # train + development only
  source_ledger.jsonl              # train + development only
  episodes/train_development.jsonl
  teacher_packets/train_development/
  sealed/                          # 只有双重确认后才存在
    fact_packs.jsonl
    source_ledger.jsonl
    episodes/sealed_test.jsonl
    teacher_packets/initial.jsonl
```

训练与开发流程只读取开放根目录。冻结评测阶段在双重确认后才读取独立 sealed 根目录。不要生成把两类 Episode、teacher packet 或 source ledger 重新合并到开放根目录的 `all_*` 文件。

## 核心约束

- 所有事实和来源不得晚于 `as_of`；
- 每包必须完整标注固定 11 个微概念；
- 非 abstain 标签恰有 1 个主模型、0–2 个辅助模型；
- `requires` 是硬前提，`modulates` 不能单独赋予主模型资格，活跃 `challenges` 不得静默忽略；
- 每题必须有合法近邻排除和下一步核验；
- 同一事实包的四个 Episode 始终处于同一切分；
- 不满足 20 主例、40 主/辅总例和 3 领域门槛的模型保持 `unsupported`。

## All-model numeric teacher scores

The current Router target is no longer restricted to one primary plus at most
two auxiliary models.  `model_score_chain_v1.py` prepares a route-blind packet
for every train/development Episode and requires each of two independent
teachers to score all 20 fixed models on an anchored 0..4 applicability scale.
The deterministic finalizer averages a model dimension only when the two scores
differ by at most the declared threshold (default: one scale point). Wider
disagreements are retained for audit and masked only on that model dimension.
The outputs remain Weak supervision rather than human Gold labels.
