---
type: microconcept
id: temporal_persistence
version: 0.1.0
status: candidate
family: state
semantic_level: microconcept
definition: 一个状态、能力或收益机制在多个时间尺度上持续而非一次性出现的程度。
mechanism: 跨连续披露期追踪方向、波动和恢复；重复出现且机制一致的信号获得更高持续性。
quantitative_evidence:
  factors:
    - {id: free_cash_flow_positive_years, role: persistence_count, transformation: count, expected_direction: positive}
    - {id: roic, role: return_quality, transformation: ratio, expected_direction: positive}
    - {id: ttm_revenue_growth, role: growth_persistence, transformation: rate, expected_direction: positive}
    - {id: earnings_volatility, role: instability, transformation: dispersion, expected_direction: negative}
    - {id: ttm_operating_margin, role: margin_persistence, transformation: ratio, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: free_cash_flow_positive_years]]', '[[EVIDENCE_CATALOG#Factor: roic]]', '[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: earnings_volatility]]', '[[EVIDENCE_CATALOG#Factor: ttm_operating_margin]]']
qualitative_evidence:
  claim_types: [revenue_demand_change, margin_pressure, reinvestment_quality, operational_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: revenue_demand_change]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: reinvestment_quality]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-compounding, relation: supports, rationale: 持续回报和再投资是复利前提}
  - {model_id: mm-p1-reliability-lifetime, relation: supports, rationale: 跨期稳定性影响寿命与失效率估计}
  - {model_id: mm-p1-stability-transition, relation: modulates, rationale: 持续恶化可提示状态转移}
counterexamples: [短历史或结构性行业变化会混淆持续与过时]
falsifiers: [新增多个披露期后信号快速回归均值]
counterfactual_tests: [增加连续正现金流期数应提高持续性, 增加波动应降低持续性]
failure_conditions: [样本期不足, 历史数据有后验修订]
evidence_boundary: 不把过去持续直接当作未来持续的保证。
---
# Temporal Persistence

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
