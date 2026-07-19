---
type: research_data_provenance
title: Phase 1 Value-Investing Data Provenance
status: active
version: 0.1.0
updated: 2026-07-19
---

# Phase 1 Value-Investing Data Provenance

本文记录当前价值投资 Episode、财报 HTML 和 MD&A 定位数据的来源、处理
边界与可复现路径。原始文件和外部市场数据不提交仓库。

## 当前数据产物

| 产物 | 外部路径 | 用途 | 当前状态 |
| --- | --- | --- | --- |
| SEC 公司和申报元数据 | `D:\LollaNetData\sec-universe-200` | 200 家候选公司的 CIK、申报列表、Company Facts | 已下载 |
| SEC 财务候选 Episode | `D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl` | 财报时点因子、市场状态、Episode 标识 | 6,549 条 |
| SEC 原始申报 HTML | `D:\LollaNetData\sec-mda-v1\filings` | 10-K/10-Q 原始文档，供本地定位和后续抽取 | 6,549 份，约 9.1 GB |
| MD&A 定位索引 | `D:\LollaNetData\sec-mda-v1\mda_index.jsonl` | accession、来源 URL、章节偏移、哈希和定位质量 | 6,549 条 |
| MD&A 下载报告 | `D:\LollaNetData\sec-mda-v1\download_report.json` | 下载和定位覆盖统计 | 当前版本 |
| 免费市场数据 | `D:\LollaNetData\yahoo-prices-200` 等 | 探索性价格特征，不具备退市收益契约 | 仅研究试验 |

## 数据来源

### SEC EDGAR

财报身份和财务因子来自 SEC EDGAR 的公开数据：

- `data.sec.gov/submissions/CIK##########.json`：公司申报元数据；
- `data.sec.gov/api/xbrl/companyfacts/CIK##########.json`：XBRL Company Facts；
- `www.sec.gov/Archives/edgar/data/...`：申报主文档 HTML；
- 申报表单：10-K、10-Q；
- 可用时间：`acceptanceDateTime`，而不是事后下载时间。

SEC 请求通过 `SEC_USER_AGENT` 或 `--user-agent` 注入真实联系信息，采用限速、
重试和缓存。下载器位于
[download_sec_mda.py](download_sec_mda.py)。

### 市场数据

当前市场字段来自免费探索性价格源和现有外部批处理。它们用于研究原型，
存在当前 ticker 历史映射、退市收益和点时证券映射限制。市场数据不能支持
研究级 alpha、完整总回报或自动交易结论。

### MD&A

MD&A 不购买第三方文本数据库，直接从 SEC 原始申报中定位：

- 10-K：Item 7；旧式/小型报告公司也支持 Item 6；
- 10-Q：Item 2；
- 优先路径：目录 `href="#..."` 到正文 `id/name` 的 DOM 锚点；
- 降级路径：标题变体和正文语义信号的文本定位；
- 章节结束：下一个 Item 3、4、7A、8 或 9 的边界。

当前版本的定位质量：

```text
Episode 总数：6,549
可定位章节：6,466
高置信度：5,818
低置信度：648
未定位：83
DOM toc_anchor：2,336
semantic_heading：1,230
legacy_item6：2
网络失败：0
```

高置信度定位要求正式 MD&A 标题、足够正文长度、至少两个经营/流动性/现金流
信号，以及可识别的下一 Item 边界。低置信度和未定位记录不会直接用于第一批
微概念教师标签。

## 财务因子处理

字段定义唯一来源是
[feature_dictionary.yaml](feature_dictionary.yaml)。当前主要字段包括：

- 收入增长、毛利变化、经营利润率；
- TTM 经营现金流、资本开支、自由现金流；
- 正自由现金流年数、ROIC；
- 净债务/EBITDA、利息覆盖、流动比率；
- 应收和库存增长相对收入增长的差值；
- 资本开支/收入变化、盈利意外、盈利波动；
- 市值、自由现金流收益率、估值分位、行业代码、回撤。

处理约束：

1. Company Facts 只使用申报可用时间不晚于 Episode 的观察；
2. 流量字段按季度重建并聚合为 TTM；必要时从年度值扣除前三个季度推导 Q4；
3. 存量字段选择不晚于申报可用时间的最新观察；
4. 比率和增长率保留原始缺失，不把缺失填成 0；
5. 每条记录保留来源字段、缺失原因和事实版本；
6. 市场字段无法获得时显式标记缺失，不伪造估值或收益。

## MD&A 处理边界

当前只保存：

- accession；
- form 和 filing date；
- public available time；
- 原始文件 URL 和外部文件路径；
- 原始 HTML SHA-256；
- 规范化文本的章节起止偏移；
- 章节 SHA-256；
- locator method、质量等级和 extractor version。

当前不保存到仓库或训练包：

- 完整 MD&A 原文；
- 未来财报内容；
- 未来收益、退市结果或未来指数成分；
- 由 ChatGPT 自由生成但没有可见证据的命题。

后续 ChatGPT 输入只从高置信度章节生成受控命题：

```text
operating_change
liquidity
capital_allocation
risk
management_outlook
```

每个命题还必须带 polarity、time_horizon、strength、accession、section、
偏移、哈希和抽取版本。

## 时间切分和泄漏边界

Episode 的事实可见边界是 `public_available_at`。输入不能使用：

- 申报接受时间之后才出现的事实修订；
- 未来价格或 12 个月回报；
- 未来 MD&A；
- 当前 ticker 对历史证券身份的未经验证替代；
- 结果字段生成的路由标签。

路由标签和 ChatGPT 教师标签是监督信号，不是输入字段。教师标签是伪标签，
不是人工金标准；未来需要人工复核冲突样本和跨学科映射。

## 可复现命令

```powershell
# 校验仓库数据契约
python research/phase1/value_investing/validate_dataset.py

# 下载并定位 SEC 原始财报（原始文件写入外部目录）
$env:SEC_USER_AGENT = "Your Name your.email@university.edu"
python research/phase1/value_investing/download_sec_mda.py `
  --input D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --raw-root D:\LollaNetData\sec-universe-200 `
  --output D:\LollaNetData\sec-mda-v1 `
  --workers 8 --delay 0.12

# 只升级低置信度缓存，不重新下载
python research/phase1/value_investing/download_sec_mda.py `
  --input D:\LollaNetData\sec-candidates-market-valued\sec_candidates_market_valued.jsonl `
  --raw-root D:\LollaNetData\sec-universe-200 `
  --output D:\LollaNetData\sec-mda-v1 `
  --workers 8 --semantic-only
```

## 当前限制

- SEC HTML 的历史格式、目录结构和 Item 编号不统一；
- 83 条记录仍未能可靠定位 MD&A；
- 648 条低置信度记录需要人工或更强的结构化抽取复核；
- 免费市场源不提供完整退市收益和可靠历史证券映射；
- 当前尚未生成真实 MD&A 结构化命题，也尚未训练微概念编码器；
- 数据集用于方法验证，不构成投资建议或自动交易系统。
