---
type: microconcept
id: constraint_pressure
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 哪个资源、能力或约束最限制系统目标的边际改善。
mechanism: 比较候选环节的缺口、利用率和对总结果的边际影响；压力集中处优先决定系统表现。
quantitative_evidence:
  factors:
    - {id: ttm_revenue_growth, role: target_flow, transformation: rate, expected_direction: positive}
    - {id: gross_margin_change, role: unit_economics_pressure, transformation: delta, expected_direction: negative}
    - {id: current_ratio, role: liquidity_constraint, transformation: ratio, expected_direction: positive}
    - {id: receivables_growth_minus_revenue_growth, role: collection_constraint, transformation: delta, expected_direction: negative}
    - {id: inventory_growth_minus_revenue_growth, role: inventory_constraint, transformation: delta, expected_direction: negative}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: gross_margin_change]]', '[[EVIDENCE_CATALOG#Factor: current_ratio]]', '[[EVIDENCE_CATALOG#Factor: receivables_growth_minus_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: inventory_growth_minus_revenue_growth]]']
qualitative_evidence:
  claim_types: [revenue_demand_change, margin_pressure, working_capital_change, operational_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: revenue_demand_change]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: working_capital_change]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-bottleneck, relation: supports, rationale: 定位限制总吞吐的环节}
  - {model_id: mm-p1-rate-control, relation: supports, rationale: 识别最慢的控制步骤}
  - {model_id: mm-p1-optimization-constraints, relation: supports, rationale: 将资源和目标写成受限优化问题}
counterexamples: [需求不足时高利用率不代表内部约束]
falsifiers: [缓解被标记环节后总产出没有边际改善]
counterfactual_tests: [放宽候选约束应提高目标或降低周期, 收紧约束应扩大压力]
failure_conditions: [系统边界不明, 多个约束同时变化且无法归因]
evidence_boundary: 不把最差财务比率自动当作真正业务瓶颈。
---
# Constraint Pressure

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
