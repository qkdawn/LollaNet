---
type: research_index
title: Phase 1 Microconcept Index
status: candidate
generated_by: research/phase1/microconcepts/generate_index.py
---

# Microconcept Index

Markdown 卡是唯一事实源；本页为生成导航。

| Microconcept | Family | Quantitative | Qualitative | Models |
| --- | --- | --- | --- | --- |
| [[capacity_headroom]] | state | current_ratio, interest_coverage, net_debt_to_ebitda, ttm_operating_cash_flow | liquidity_outlook, refinancing_dependency, covenant_risk | mm-p1-buffer-capacity, mm-p1-safety-margin, mm-p1-reliability-lifetime |
| [[constraint_pressure]] | relation | ttm_revenue_growth, gross_margin_change, current_ratio, receivables_growth_minus_revenue_growth, inventory_growth_minus_revenue_growth | revenue_demand_change, margin_pressure, working_capital_change, operational_risk | mm-p1-bottleneck, mm-p1-rate-control, mm-p1-optimization-constraints |
| [[conversion_gap]] | relation | ttm_revenue_growth, ttm_operating_margin, ttm_operating_cash_flow, ttm_free_cash_flow, receivables_growth_minus_revenue_growth, inventory_growth_minus_revenue_growth | working_capital_change, margin_pressure, capital_allocation_change | mm-p1-bottleneck, mm-p1-feedback-loop, mm-p1-mass-stoichiometry |
| [[countervailing_pressure]] | relation | ttm_revenue_growth, gross_margin_change, ttm_capex, net_debt_to_ebitda, interest_coverage | revenue_demand_change, margin_pressure, capital_allocation_change, covenant_risk | mm-p1-force-balance, mm-p1-equilibrium-response, mm-p1-feedback-loop |
| [[evidence_quality]] | evidence | earnings_surprise_zscore, earnings_volatility | guidance_revision, liquidity_outlook, operational_risk, demand_outlook | mm-p1-numerical-error, mm-p1-bayesian-update, mm-p1-local-sensitivity |
| [[feedback_polarity]] | relation | ttm_revenue_growth, gross_margin_change, ttm_free_cash_flow, ttm_capex, roic | reinvestment_quality, capital_allocation_change, operational_risk, demand_outlook | mm-p1-feedback-loop, mm-p1-equilibrium-response, mm-p1-stability-transition |
| [[outcome_asymmetry]] | state | free_cash_flow_yield, valuation_percentile, earnings_volatility, trailing_drawdown, interest_coverage | demand_outlook, liquidity_outlook, operational_risk, guidance_revision | mm-p1-expected-value, mm-p1-opportunity-cost, mm-p1-safety-margin |
| [[regime_proximity]] | state | current_ratio, interest_coverage, net_debt_to_ebitda, earnings_volatility, gross_margin_change | liquidity_outlook, covenant_risk, margin_pressure, operational_risk | mm-p1-stability-transition, mm-p1-safety-margin, mm-p1-buffer-capacity |
| [[relative_gradient]] | relation | ttm_revenue_growth, gross_margin_change, earnings_surprise_zscore, valuation_percentile, trailing_drawdown | revenue_demand_change, margin_pressure, guidance_revision, demand_outlook | mm-p1-diffusion-gradient, mm-p1-opportunity-cost, mm-p1-local-sensitivity |
| [[resource_balance]] | relation | ttm_operating_cash_flow, ttm_capex, ttm_free_cash_flow, ttm_revenue_growth | working_capital_change, capital_allocation_change, operational_risk | mm-p1-mass-stoichiometry, mm-p1-energy-conservation, mm-p1-function-mapping |
| [[temporal_persistence]] | state | free_cash_flow_positive_years, roic, ttm_revenue_growth, earnings_volatility, ttm_operating_margin | revenue_demand_change, margin_pressure, reinvestment_quality, operational_risk | mm-p1-compounding, mm-p1-reliability-lifetime, mm-p1-stability-transition |
