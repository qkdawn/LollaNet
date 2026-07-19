---
type: microconcept
id: relative_gradient
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 状态、资源或机会相对参考点的差距、方向和变化速度。
mechanism: 选定基准后计算差值或斜率；梯度方向提示资源、价格或行动的迁移方向。
quantitative_evidence:
  factors:
    - {id: ttm_revenue_growth, role: operating_gradient, transformation: rate, expected_direction: positive}
    - {id: gross_margin_change, role: margin_gradient, transformation: delta, expected_direction: positive}
    - {id: earnings_surprise_zscore, role: evidence_gradient, transformation: standardized_delta, expected_direction: positive}
    - {id: valuation_percentile, role: relative_valuation, transformation: percentile, expected_direction: negative}
    - {id: trailing_drawdown, role: price_state, transformation: drawdown, expected_direction: mixed}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: gross_margin_change]]', '[[EVIDENCE_CATALOG#Factor: earnings_surprise_zscore]]', '[[EVIDENCE_CATALOG#Factor: valuation_percentile]]', '[[EVIDENCE_CATALOG#Factor: trailing_drawdown]]']
qualitative_evidence:
  claim_types: [revenue_demand_change, margin_pressure, guidance_revision, demand_outlook]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: revenue_demand_change]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: guidance_revision]]', '[[EVIDENCE_CATALOG#Claim: demand_outlook]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-diffusion-gradient, relation: supports, rationale: 表示相对状态差及其趋向}
  - {model_id: mm-p1-opportunity-cost, relation: supports, rationale: 比较候选行动与基准的相对收益}
  - {model_id: mm-p1-local-sensitivity, relation: modulates, rationale: 梯度可作为局部变化方向}
counterexamples: [基准选择错误会制造虚假梯度]
falsifiers: [更换合理基准后方向消失或反转]
counterfactual_tests: [提高相对估值分位应改变机会差, 提高意外收益应改变证据梯度]
failure_conditions: [无稳定基准, 混用不同时间尺度]
evidence_boundary: 不把相对差距解释成必然的因果流动。
---
# Relative Gradient

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
