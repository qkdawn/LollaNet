---
type: microconcept
id: evidence_quality
version: 0.1.0
status: candidate
family: evidence
semantic_level: microconcept
definition: 当前证据对目标判断的可靠性、独立性、及时性和可重复程度。
mechanism: 分离可用时间、来源层级、缺失、修订、噪声和相互依赖，限制结论置信度而非替代结论。
quantitative_evidence:
  factors:
    - {id: earnings_surprise_zscore, role: signal_strength, transformation: standardized_delta, expected_direction: positive}
    - {id: earnings_volatility, role: noise, transformation: dispersion, expected_direction: negative}
  links: ['[[EVIDENCE_CATALOG#Factor: earnings_surprise_zscore]]', '[[EVIDENCE_CATALOG#Factor: earnings_volatility]]']
qualitative_evidence:
  claim_types: [guidance_revision, liquidity_outlook, operational_risk, demand_outlook]
  aggregation: {direction: signed_claims, missing_policy: preserve_unknown}
  links: ['[[EVIDENCE_CATALOG#Claim: guidance_revision]]', '[[EVIDENCE_CATALOG#Claim: liquidity_outlook]]', '[[EVIDENCE_CATALOG#Claim: operational_risk]]', '[[EVIDENCE_CATALOG#Claim: demand_outlook]]']
output_interface: {presence: probability, intensity: signed_numeric, direction: categorical, confidence: probability}
model_relations:
  - {model_id: mm-p1-numerical-error, relation: supports, rationale: 传播缺失、误差和修订不确定性}
  - {model_id: mm-p1-bayesian-update, relation: modulates, rationale: 决定证据对后验的权重}
  - {model_id: mm-p1-local-sensitivity, relation: challenges, rationale: 检查结论是否由单个不稳健字段主导}
counterexamples: [高质量来源也可能对目标假设缺乏区分力]
falsifiers: [独立来源和后续修订显示当前证据具有系统性偏差]
counterfactual_tests: [降低字段置信度应降低后验更新幅度, 移除单一字段后结论应检验是否稳定]
failure_conditions: [来源血缘缺失, 可用时间晚于决策时间, 同源证据重复计数]
evidence_boundary: 证据质量高不等于命题为真，只表示可依赖程度更高。
---
# Evidence Quality

[[EVIDENCE_CATALOG]] · [[MICROCONCEPT_INDEX]]
