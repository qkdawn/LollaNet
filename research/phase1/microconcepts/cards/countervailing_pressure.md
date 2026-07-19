---
type: microconcept
id: countervailing_pressure
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 同一状态变量上方向相反的作用及其净效应是否接近平衡。
mechanism: 将推动与抑制因素映射到同一目标和时间尺度，比较方向、强度与响应滞后而非简单计数。
quantitative_evidence:
  factors:
    - {id: ttm_revenue_growth, role: growth_pressure, transformation: rate, expected_direction: positive}
    - {id: gross_margin_change, role: cost_pressure, transformation: delta, expected_direction: positive}
    - {id: ttm_capex, role: investment_load, transformation: ttm_flow, expected_direction: negative}
    - {id: net_debt_to_ebitda, role: financing_load, transformation: ratio, expected_direction: negative}
    - {id: interest_coverage, role: debt_capacity, transformation: ratio, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: gross_margin_change]]', '[[EVIDENCE_CATALOG#Factor: ttm_capex]]', '[[EVIDENCE_CATALOG#Factor: net_debt_to_ebitda]]', '[[EVIDENCE_CATALOG#Factor: interest_coverage]]']
qualitative_evidence:
  claim_types: [revenue_demand_change, margin_pressure, capital_allocation_change, covenant_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: revenue_demand_change]]', '[[EVIDENCE_CATALOG#Claim: margin_pressure]]', '[[EVIDENCE_CATALOG#Claim: capital_allocation_change]]', '[[EVIDENCE_CATALOG#Claim: covenant_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-force-balance, relation: requires, rationale: 把增长、成本和融资压力投影到共同目标上}
  - {model_id: mm-p1-equilibrium-response, relation: supports, rationale: 净压力决定扰动后状态移动方向}
  - {model_id: mm-p1-feedback-loop, relation: modulates, rationale: 反馈会放大或抵消初始净压力}
counterexamples: [不同单位的财务变化不能直接相减为净力]
falsifiers: [被判断为占优的作用增强后目标状态未按预测方向移动]
counterfactual_tests: [提高利息负担应增加负向压力, 毛利改善而其他条件不变应增加正向压力]
failure_conditions: [作用对象不同, 时间滞后不一致, 无法定义共同响应变量]
evidence_boundary: 力是跨域机制映射，不声称财务系统服从物理力学方程。
---
# Countervailing Pressure

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
