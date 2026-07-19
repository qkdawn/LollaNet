---
type: microconcept
id: outcome_asymmetry
version: 0.1.0
status: candidate
family: state
semantic_level: microconcept
definition: 候选结果在上行、下行幅度和发生可能性上是否呈现不对称结构。
mechanism: 分开表示情景概率与损益幅度，识别均值相似但尾部代价、可逆性或上行空间不同的选择。
quantitative_evidence:
  factors:
    - {id: free_cash_flow_yield, role: cash_valuation, transformation: ratio, expected_direction: positive}
    - {id: valuation_percentile, role: relative_valuation, transformation: percentile, expected_direction: negative}
    - {id: earnings_volatility, role: outcome_dispersion, transformation: dispersion, expected_direction: negative}
    - {id: trailing_drawdown, role: observed_downside, transformation: drawdown, expected_direction: negative}
    - {id: interest_coverage, role: downside_capacity, transformation: ratio, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: free_cash_flow_yield]]', '[[EVIDENCE_CATALOG#Factor: valuation_percentile]]', '[[EVIDENCE_CATALOG#Factor: earnings_volatility]]', '[[EVIDENCE_CATALOG#Factor: trailing_drawdown]]', '[[EVIDENCE_CATALOG#Factor: interest_coverage]]']
qualitative_evidence:
  claim_types: [demand_outlook, liquidity_outlook, operational_risk, guidance_revision]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: demand_outlook]]', '[[EVIDENCE_CATALOG#Claim: liquidity_outlook]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]', '[[EVIDENCE_CATALOG#Claim: guidance_revision]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-expected-value, relation: requires, rationale: 期望比较需要同时保留概率和结果幅度}
  - {model_id: mm-p1-opportunity-cost, relation: supports, rationale: 候选方案的分布差异改变放弃另一方案的代价}
  - {model_id: mm-p1-safety-margin, relation: modulates, rationale: 下行尾部越重所需安全余量越大}
counterexamples: [历史波动低不表示未来下行尾部较轻]
falsifiers: [加入可辩护的新情景后不对称方向消失或反转]
counterfactual_tests: [扩大下行情景损失应降低净吸引力, 降低下行情景概率应减弱负向不对称]
failure_conditions: [情景集合遗漏主要结果, 概率与损益来自不同时间尺度]
evidence_boundary: 不把历史价格分布直接当作未来基本面结果分布。
---
# Outcome Asymmetry

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
