---
type: microconcept
id: feedback_polarity
version: 0.1.0
status: candidate
family: relation
semantic_level: microconcept
definition: 结果变化通过反馈回路强化、抵消还是延迟了后续变化。
mechanism: 追踪行动、产出、现金、再投资和下一期结果的有向滞后关系，判断反馈符号与增益。
quantitative_evidence:
  factors:
    - {id: ttm_revenue_growth, role: output_flow, transformation: rate, expected_direction: positive}
    - {id: gross_margin_change, role: operating_response, transformation: delta, expected_direction: positive}
    - {id: ttm_free_cash_flow, role: reinvestment_resource, transformation: derived_flow, expected_direction: positive}
    - {id: ttm_capex, role: intervention, transformation: ttm_flow, expected_direction: mixed}
    - {id: roic, role: reinvestment_return, transformation: ratio, expected_direction: positive}
  links: ['[[EVIDENCE_CATALOG#Factor: ttm_revenue_growth]]', '[[EVIDENCE_CATALOG#Factor: gross_margin_change]]', '[[EVIDENCE_CATALOG#Factor: ttm_free_cash_flow]]', '[[EVIDENCE_CATALOG#Factor: ttm_capex]]', '[[EVIDENCE_CATALOG#Factor: roic]]']
qualitative_evidence:
  claim_types: [reinvestment_quality, capital_allocation_change, operational_risk, demand_outlook]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: reinvestment_quality]]', '[[EVIDENCE_CATALOG#Claim: capital_allocation_change]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]', '[[EVIDENCE_CATALOG#Claim: demand_outlook]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-feedback-loop, relation: supports, rationale: 判断再投资和经营结果的反馈符号}
  - {model_id: mm-p1-equilibrium-response, relation: supports, rationale: 观察扰动后回到或偏离平衡的方向}
  - {model_id: mm-p1-stability-transition, relation: modulates, rationale: 正反馈增益可能推近临界状态}
counterexamples: [短样本中的相关性可能只是共同外部冲击]
falsifiers: [跨多个披露期反馈方向不稳定或加入外部变量后消失]
counterfactual_tests: [提高高回报再投资比例应强化后续增长, 降低毛利应削弱正反馈]
failure_conditions: [时间滞后未知, 不能区分共同原因与反馈]
evidence_boundary: 不从两期相关变化直接断言存在因果反馈。
---
# Feedback Polarity

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
