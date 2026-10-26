---
type: case
case_name: "mqv1-sp-016-usgs-us6000k49j"
domain: "Public-Systems"
created: 2026-07-20
updated: 2026-07-20
date_observed: "2023-04-14"
as_of: 2026-07-20
privacy: public
provenance: public-sourced
verification_status: sourced
public_case: true
source_refs:
  - "https://earthquake.usgs.gov/earthquakes/eventpage/us6000k49j"
  - "https://earthquake.usgs.gov/fdsnws/event/1/query"
related_models: []
tags:
  - case
  - mechanism-question-v1
  - science_public_systems
---

# mqv1-sp-016-usgs-us6000k49j

## 案例摘要

本卡是机制问题集 V1 的公开事实摘要。正文使用中性案例编号；真实机构或产品身份只保留在来源定位中，不作为路由捷径。

## 背景

该记录来自公开、结构化的官方数据源，用于训练“事实—微概念状态—机制选择”接口。案例只保存回答问题所需的最小事实，不复制原始页面全文。

## 时代与制度条件

- 本卡按 2026-07-20T00:00:00Z 的公开数据结构与口径整理。数据源后续修订、报告覆盖率、制度要求和测量方法变化都可能改变可比性。
- 跨领域迁移只在压力、剩余空间、证据质量、转换差距或失效代价等结构可回引时成立；对象名称相似不构成证据。

## 关键事实

1. USGS 将事件 us6000k49j 标记为 reviewed；发生时间为 2023-04-14，震级为 7.0，深度约 597.0 千米。
2. 事件的仪器烈度为 2.78，影响警报等级为 green，显著性指标为 827。
3. 公众感知报告数为 133，台站数为 203，方位覆盖缺口为 31 度，残差指标为 0.71。
4. USGS 记录的海啸标志为 0；事件坐标为纬度 -6.0413、经度 112.0478。

## 事实、推断与待验证项

| 类型 | 内容 |
| --- | --- |
| 事实 | 上述关键事实均回引到本卡来源，并受资料截点约束。 |
| 整理者推断 | 这些事实可以支持或挑战若干可迁移机制状态，但不能单独证明任何正式思维模型为真。 |
| 待验证 | - 单个事件记录不包含完整暴露人口、基础设施脆弱性、次生灾害和恢复过程，不能由震级直接推出社会损失。 |

## 涉及模型

- 不在公共案例卡中写入教师路由。模型选择标签隔离保存在外部 Weak 标签目录。

## 模型解释

本卡只提供事实与机制状态的可审计基础。具体主模型、辅助模型和近邻排除由路由阶段根据固定关系索引生成，不反向修改本卡事实。

## 失败机制或关键转折

当前公开记录只允许识别压力、边界、转换、证据或资源关系的候选结构；若关键状态为 unknown，必须保留未知，不能补写一个确定性失败故事。

## 耐久机制

可迁移部分是可观察变量之间的结构关系及其失效条件，而不是机构名称、产品类别或历史结果本身。

## 竞争性解释与反证

- 同一组事实可能同时支持多个相邻机制，需用产生不同预测的新证据区分。
- 比率、期限和严重度字段不是对机构质量、责任或未来结果的总评。
- 若来源修订、口径不可比或关键变量缺失，当前状态应降级或 abstain。

## 来源

- USGS reviewed event us6000k49j：https://earthquake.usgs.gov/earthquakes/eventpage/us6000k49j（定位："us6000k49j"）
- USGS FDSN event query：https://earthquake.usgs.gov/fdsnws/event/1/query（定位：{"format": "geojson", "starttime": "2023-01-01", "endtime": "2023-12-31", "minmagnitude": 5.5, "orderby": "magnitude", "limit": 200}）

本卡只概括公开记录中的必要字段，不复制原始网页或报告全文；教师路由标签不写入公共案例库。

## 模型修改提案

暂不提出。该记录用于弱监督路由研究，不足以单独修改正式模型语义。
