---
type: microconcept
id: capacity_headroom
version: 0.1.0
status: candidate
family: state
semantic_level: microconcept
definition: 当前资源或能力距离耗尽、违约或不可接受失效边界的剩余空间。
mechanism: 将可用存量、持续流出速率和约束阈值放在同一单位下，余量随消耗上升而下降。
quantitative_evidence:
  factors:
    - {id: current_ratio, role: liquidity_capacity, transformation: point_in_time_ratio, expected_direction: positive}
    - {id: interest_coverage, role: debt_service_capacity, transformation: point_in_time_ratio, expected_direction: positive}
    - {id: net_debt_to_ebitda, role: leverage_load, transformation: ratio, expected_direction: negative}
    - {id: ttm_operating_cash_flow, role: replenishment_flow, transformation: ttm_flow, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: current_ratio]]', '[[EVIDENCE_CATALOG#Factor: interest_coverage]]', '[[EVIDENCE_CATALOG#Factor: net_debt_to_ebitda]]', '[[EVIDENCE_CATALOG#Factor: ttm_operating_cash_flow]]']
qualitative_evidence:
  claim_types: [liquidity_outlook, refinancing_dependency, covenant_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: liquidity_outlook]]', '[[EVIDENCE_CATALOG#Claim: refinancing_dependency]]', '[[EVIDENCE_CATALOG#Claim: covenant_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-buffer-capacity, relation: supports, rationale: 量化可吸收扰动的容量}
  - {model_id: mm-p1-safety-margin, relation: supports, rationale: 评估距风险边界的距离}
  - {model_id: mm-p1-reliability-lifetime, relation: modulates, rationale: 余量决定给定负荷下的失效时间}
counterexamples: [受监管行业的高杠杆可能是制度常态]
falsifiers: [后续现金转换和债务偿还显示余量没有按预期恶化]
counterfactual_tests: [降低 current_ratio 时余量应下降, 提高 interest_coverage 时余量应上升]
failure_conditions: [容量单位不可比, 负荷或阈值未知]
evidence_boundary: 不把当前比率直接等同于未来违约概率。
---
# Capacity Headroom

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
