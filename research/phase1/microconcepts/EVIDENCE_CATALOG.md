---
type: research_evidence_catalog
title: Phase 1 Evidence Catalog
status: candidate
generated_by: research/phase1/microconcepts/generate_index.py
---

# Evidence Catalog

此页由字段字典和受控命题目录生成，只用于导航。

## Quantitative Factors

### Factor: capex_to_revenue_change

- Unit: `ratio`
- Source: derived
- Formula: current capex/revenue minus prior capex/revenue

### Factor: current_ratio

- Unit: `ratio`
- Source: point-in-time balance sheet
- Formula: current assets divided by current liabilities

### Factor: earnings_surprise_zscore

- Unit: `numeric`
- Source: derived from previously filed SEC XBRL earnings
- Formula: standardized reported earnings change versus trailing reported history

### Factor: earnings_volatility

- Unit: `ratio`
- Source: derived
- Formula: standard deviation of quarterly earnings over trailing twelve quarters

### Factor: free_cash_flow_positive_years

- Unit: `count`
- Source: derived
- Formula: positive annual free-cash-flow count over prior three fiscal years

### Factor: free_cash_flow_yield

- Unit: `ratio`
- Source: derived
- Formula: TTM free cash flow divided by market capitalization

### Factor: gross_margin_change

- Unit: `ratio`
- Source: SEC XBRL facts filtered by filing date
- Formula: current TTM gross margin minus prior TTM gross margin

### Factor: interest_coverage

- Unit: `ratio`
- Source: derived
- Formula: TTM EBIT divided by TTM interest expense

### Factor: inventory_growth_minus_revenue_growth

- Unit: `ratio`
- Source: derived
- Formula: inventory growth minus revenue growth

### Factor: market_cap

- Unit: `currency`
- Source: public daily close and SEC XBRL shares outstanding
- Formula: entry-date close times point-in-time shares outstanding

### Factor: net_debt_to_ebitda

- Unit: `ratio`
- Source: derived
- Formula: cash-adjusted debt divided by TTM EBITDA

### Factor: operating_cash_flow_sign_changed

- Unit: `categorical`
- Source: derived
- Formula: whether TTM operating cash flow changed sign versus the prior TTM

### Factor: receivables_growth_minus_revenue_growth

- Unit: `ratio`
- Source: derived
- Formula: receivables growth minus revenue growth

### Factor: roic

- Unit: `ratio`
- Source: derived
- Formula: NOPAT divided by invested capital

### Factor: sector_code

- Unit: `categorical`
- Source: point-in-time classification
- Formula: provider sector classification at availability time

### Factor: trailing_drawdown

- Unit: `ratio`
- Source: public daily closes
- Formula: drawdown over the prior 252 trading days

### Factor: ttm_capex

- Unit: `currency`
- Source: SEC XBRL cash flow facts
- Formula: sum of four reported fiscal quarters

### Factor: ttm_free_cash_flow

- Unit: `currency`
- Source: derived
- Formula: operating cash flow minus capital expenditure

### Factor: ttm_operating_cash_flow

- Unit: `currency`
- Source: SEC XBRL cash flow facts
- Formula: sum of four reported fiscal quarters

### Factor: ttm_operating_margin

- Unit: `ratio`
- Source: SEC XBRL facts filtered by filing date
- Formula: operating income divided by revenue

### Factor: ttm_revenue_growth

- Unit: `ratio`
- Source: SEC XBRL facts filtered by filing date
- Formula: TTM revenue growth versus prior TTM

### Factor: valuation_percentile

- Unit: `percentile`
- Source: derived
- Formula: sector-relative value percentile using only current snapshot data

## Qualitative Claims

### Claim: capital_allocation_change

- Family: `capital_allocation`
- Definition: 管理层披露资本开支、回购、分红或并购变化
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: covenant_risk

- Family: `liquidity`
- Definition: 债务契约或偿付约束风险
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: demand_outlook

- Family: `management_outlook`
- Definition: 管理层对未来需求或经营趋势的展望
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: guidance_revision

- Family: `management_outlook`
- Definition: 指引相对历史或前期的变化
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: liquidity_outlook

- Family: `liquidity`
- Definition: 管理层对未来流动性的判断
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: margin_pressure

- Family: `operating_change`
- Definition: 管理层披露毛利或经营利润压力
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: operational_risk

- Family: `risk`
- Definition: 运营、供应链或执行风险披露
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: refinancing_dependency

- Family: `liquidity`
- Definition: 对未来融资或再融资的依赖
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: reinvestment_quality

- Family: `capital_allocation`
- Definition: 再投资方向和回报质量的管理层证据
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: revenue_demand_change

- Family: `operating_change`
- Definition: 管理层披露需求或收入变化
- Locator: accession, section, start/end offset, SHA-256, extractor version

### Claim: working_capital_change

- Family: `operating_change`
- Definition: 管理层披露应收、库存或营运资本变化
- Locator: accession, section, start/end offset, SHA-256, extractor version
