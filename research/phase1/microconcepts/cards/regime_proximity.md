---
type: microconcept
id: regime_proximity
version: 0.1.0
status: candidate
family: state
semantic_level: microconcept
definition: 当前系统状态距离另一种稳定、脆弱或不可逆运行状态的接近程度。
mechanism: 结合余量、波动、恢复速度和阈值附近的非线性变化，识别可能的状态转换前区间。
quantitative_evidence:
  factors:
    - {id: current_ratio, role: liquidity_margin, transformation: ratio, expected_direction: negative}
    - {id: interest_coverage, role: debt_service_margin, transformation: ratio, expected_direction: negative}
    - {id: net_debt_to_ebitda, role: leverage_load, transformation: ratio, expected_direction: positive}
    - {id: earnings_volatility, role: instability, transformation: dispersion, expected_direction: positive}
    - {id: gross_margin_change, role: deterioration_rate, transformation: delta, expected_direction: negative}
  links: ['[[EVIDENCE_CATALOG#Factor: current_ratio]]', '[[EVIDENCE_CATALOG#Factor: interest_coverage]]', '[[EVIDENCE_CATALOG#Factor: net_debt_to_ebitda]]', '[[EVIDENCE_CATALOG#Factor: earnings_volatility]]', '[[EVIDENCE_CATALOG#Factor: gross_margin_change]]']
qualitative_evidence:
  claim_types: [liquidity_outlook, covenant_risk, margin_pressure, operational_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: liquidity_outlook]]', '[[EVIDENCE_CATALOG#Claim: covenant_risk]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-stability-transition, relation: supports, rationale: 判断是否接近临界状态}
  - {model_id: mm-p1-safety-margin, relation: supports, rationale: 将临界距离转化为安全缓冲}
  - {model_id: mm-p1-buffer-capacity, relation: requires, rationale: 需要容量余量解释接近程度}
counterexamples: [比率变化可能来自会计重分类而非状态转移]
falsifiers: [改善余量后没有出现预期的恢复或风险下降]
counterfactual_tests: [降低余量并提高波动应提高接近度, 增加缓冲应降低接近度]
failure_conditions: [临界阈值未知, 外部冲击主导状态变化]
evidence_boundary: 不把临界接近概率当作违约或崩溃概率。
---
# Regime Proximity

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
