---
type: research_pipeline
title: Real MD&A Structured Claims v1
status: active
version: 1.0.0
updated: 2026-07-19
---

# 真实 MD&A 结构化命题集 v1

本目录只保存抽取与校验代码、数据契约和可复现说明；真实标签、检查点、审阅队列、原始 SEC HTML 和运行报告均在 D:\\LollaNetData，绝不提交到仓库。

## 范围与边界

- 输入索引：D:\\LollaNetData\\sec-mda-v1\\mda_index.jsonl；首批仅选择 status=located 且 locator_quality=high 的 5,818 条 MD&A locator。
- 分层：public_available_at 年份、10-K/10-Q 表单、SIC 两位行业分组。采样完全由固定 seed 和索引元数据决定。
- 原文：运行时仅从 locator 的 raw_path 读取并用既有 normalize_html 还原规范化文本；不复制原文到仓库或 claims.jsonl。
- 输入隔离：不将发行人名称、财务特征、市场状态、路由标签、模型预测或未来收益输入抽取器。
- 标签：仅允许 research/phase1/microconcepts/qualitative_claim_types.yaml 中定义的 claim_id。每条标签以绝对字符区间、章节哈希和 evidence 哈希回引原文。

## 构建


e.g.

    python research/phase1/value_investing/build_mda_claims_v1.py ^
      --candidates D:\\LollaNetData\\sec-candidates-market-valued\\sec_candidates_market_valued.jsonl ^
      --index D:\\LollaNetData\\sec-mda-v1\\mda_index.jsonl ^
      --raw-root D:\\LollaNetData\\sec-mda-v1\\filings ^
      --output D:\\LollaNetData\\mda-claims-v1 ^
      --sample-size 400

sample_index.jsonl 和 work/episodes/*.json 都是原子化、可恢复的外部检查点。相同配置的重跑会复用 source-fingerprint 一致的检查点；配置变化需要新的输出目录或 --no-resume。

## 校验

    python research/phase1/value_investing/validate_mda_claims_v1.py ^
      --claims D:\\LollaNetData\\mda-claims-v1\\claims.jsonl ^
      --index D:\\LollaNetData\\sec-mda-v1\\mda_index.jsonl ^
      --output D:\\LollaNetData\\mda-claims-v1\\validation_report.json

校验拒绝 taxonomy 违规、缺失点时可用时间、locator/raw 哈希不一致、章节外证据区间、evidence 哈希不一致、非确定性 ID，以及标签中的原文/路由/结果字段。review_queue.jsonl 显式记录 abstain 与待复核情况，不臆造命题。

## 产物

- claims.jsonl：仅含可回引的受控伪标签；
- manifest.json：来源、覆盖、抽样分层和分布；
- validation_report.json：schema/hash/locator/time 校验及确定性随机回引抽查；
- review_queue.jsonl：明确 abstain、冲突和低置信度项；
- DATASET_NOTES.md：数据来源、标签分布、错误类型和限制说明。

本流程不训练微概念编码器；标签是试点伪标签，不是人工金标准、投资建议或投资结果输入。
