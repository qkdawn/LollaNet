---
type: microconcept
id: resource_balance
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 资源流入、流出与存量变化是否在同一边界和时间尺度上相互一致。
mechanism: 对齐期初存量、期间流量和期末存量；持续缺口意味着消耗、漏损或口径不一致。
quantitative_evidence:
  factors:
    - {id: ttm_operating_cash_flow, role: inflow, transformation: ttm_flow, expected_direction: positive}
    - {id: ttm_capex, role: outflow, transformation: ttm_flow, expected_direction: negative}
    - {id: ttm_free_cash_flow, role: residual, transformation: derived_flow, expected_direction: positive}
    - {id: ttm_revenue_growth, role: output_flow, transformation: rate, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_operating_cash_flow]]', '[[EVIDENCE_CATALOG#Factor: ttm_capex]]', '[[EVIDENCE_CATALOG#Factor: ttm_free_cash_flow]]', '[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]']
qualitative_evidence:
  claim_types: [working_capital_change, capital_allocation_change, operational_risk]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: working_capital_change]]', '[[EVIDENCE_CATALOG#Claim: capital_allocation_change]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-mass-stoichiometry, relation: supports, rationale: 比较资源流入流出与存量变化}
  - {model_id: mm-p1-energy-conservation, relation: supports, rationale: 检查投入资源是否在边界内守恒或转化}
  - {model_id: mm-p1-function-mapping, relation: modulates, rationale: 为输入到输出的映射提供状态变量}
counterexamples: [现金流季节性或并购使单期存量变化不能代表结构性缺口]
falsifiers: [多期现金、经营现金流和资本开支无法重构出稳定关系]
counterfactual_tests: [提高经营现金流而保持资本开支不变时缺口应收窄, 提高资本开支时自由现金流应下降]
failure_conditions: [流量与存量期间不一致, 重大非现金项目未识别]
evidence_boundary: 不把自由现金流为正直接等同于经济利润或安全投资。
---
# Resource Balance

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
