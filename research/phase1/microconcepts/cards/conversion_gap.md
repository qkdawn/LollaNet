---
type: microconcept
id: conversion_gap
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 投入、会计利润或增长转化为可兑现现金和能力的缺口。
mechanism: 对比收入、利润、营运资本、资本开支和经营现金流的变化，区分产出增长与兑现质量。
quantitative_evidence:
  factors:
    - {id: ttm_revenue_growth, role: growth_input, transformation: rate, expected_direction: negative}
    - {id: ttm_operating_margin, role: accounting_output, transformation: ratio, expected_direction: positive}
    - {id: ttm_operating_cash_flow, role: cash_output, transformation: ttm_flow, expected_direction: positive}
    - {id: ttm_free_cash_flow, role: cash_output_after_investment, transformation: derived_flow, expected_direction: positive}
    - {id: receivables_growth_minus_revenue_growth, role: collection_gap, transformation: delta, expected_direction: negative}
    - {id: inventory_growth_minus_revenue_growth, role: inventory_gap, transformation: delta, expected_direction: negative}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: ttm_operating_margin]]', '[[EVIDENCE_CATALOG#Factor: ttm_operating_cash_flow]]', '[[EVIDENCE_CATALOG#Factor: ttm_free_cash_flow]]', '[[EVIDENCE_CATALOG#Factor: receivables_growth_minus_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: inventory_growth_minus_revenue_growth]]']
qualitative_evidence:
  claim_types: [working_capital_change, margin_pressure, capital_allocation_change]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: working_capital_change]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: capital_allocation_change]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-bottleneck, relation: supports, rationale: 兑现环节可能限制增长质量}
  - {model_id: mm-p1-feedback-loop, relation: modulates, rationale: 现金兑现会改变再投资和后续产出}
  - {model_id: mm-p1-mass-stoichiometry, relation: supports, rationale: 检查投入和产出是否守恒转化}
counterexamples: [快速增长阶段应收上升可能是正常结算周期]
falsifiers: [后续现金回收和利润质量与缺口预期相反]
counterfactual_tests: [降低应收增长差应改善现金转换, 提高资本开支而无产出时缺口应扩大]
failure_conditions: [行业结算模式未知, 营运资本受一次性事件影响]
evidence_boundary: 不把会计利润与现金转换差异直接判定为欺诈。
---
# Conversion Gap

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
