# Configuration & Data-Source Reference

This document explains **every value that appears in a generated DPR** — what it does
and **where it comes from**. It is the companion to [flow.md](flow.md) (the request
trace) and [README.md](../README.md) (the overview).

---

## 1. The five sources of every value

Every number, label, and paragraph in a DPR originates from exactly one of these:

| Source | What it provides | Where it lives |
|---|---|---|
| **① Form payload** | The FPO's own inputs — identity, capex, capacity, price, manpower, loan, scheme, escalation | the website `fields` object |
| **② Business-line template** | Technical norms + default financial config for that line | `data/templates/<line>.json` |
| **③ Line defaults** | Per-line fallback for capacity / price / raw-material when the form leaves them blank | `LINE_DEFAULTS` in [form_to_instance.py](../dpr/form_to_instance.py) |
| **④ Engine** | *All* computed financial figures (cost, loan, WC, depreciation, P&L, DSCR, BEP) | [engine/financial_engine.py](../dpr/engine/financial_engine.py) |
| **⑤ LLM** | Only the §2 narrative prose (product, market, value chain) | finetuned gemma-4-12B (v4) |

**The two-lane rule:** ① ② ③ feed the engine (④), which owns **every number**; the
LLM (⑤) writes **only prose** and never a figure. `verify()` rejects any number the
LLM types that doesn't match an engine value.

---

## 2. Resolution order — "where did this config come from?"

For any configurable value the pipeline resolves it in this order:

```
form payload value  →  (if blank/invalid)  template default  →  (if none)  line default
```

In code this is `config = { **template_config, **form_overrides }` — the template
supplies the baseline, and any field the FPO filled in overrides it. So:

- A value the FPO typed → **① form**.
- A value the FPO left blank that the template carries → **② template**.
- Capacity / price / raw-material left blank (templates don't carry these) → **③ line default**.

---

## 3. Form fields — what each does & where it lands

| Form field | What it does | Lands in (instance) | Source |
|---|---|---|---|
| `fpoName`, `location`, `members` | Identity shown in §1 / narrative | `narrative.*` | ① |
| `registrationNo`, `ceoName`, `mobile`, `email`, `siteAddress` | Applicant/contact block in §1 | `narrative.*` | ① |
| `year`, `turnover`, `crops` | Governance/context text (narrative only) | `narrative.governance`, context | ① |
| `businessLine` | Selects the template (one of 12 lines) | `meta` + template load | ① → ② |
| `commodity` / `commodityOther` | Names the product; for Oil/Juice/Spice the LLM writes a product-specific intro | `meta.commodity`, report title | ① |
| `machinery[]` `{item, cost_inr}` | Plant & machinery cost — **engine sums the rows** | `inputs.capex.machinery_inr` | ① (blank → ② template list) |
| `civilCost` | Civil works — **always capitalised** (any land status) | `inputs.capex.civil_construction_lakh` | ① |
| `otherFixedAssets` | Misc fixed assets | `inputs.capex.misc_fixed_assets_lakh` | ① (blank → ② `technical_defaults`) |
| `landStatus`, `monthlyRent` | Owned vs leased; rent charged to P&L if leased | `inputs.land_status`, `rent_per_month_inr` | ① |
| `capacity`, `workingDays` | Throughput per day (service lines = annual) & operating days | `inputs.capacity.*` | ① (blank → ③) |
| `salePrice` | Sale price per unit | `inputs.prices.sale_price` | ① (blank → ③) |
| `rmcost` `{mode, value}` | Raw-material cost: `unit` = per-unit rate, `pct` = % of revenue | `inputs.prices.*` | ① (blank → ③) |
| `manpower[]` `{role, count, monthly_wage_inr}` | Wage roster — **split** Worker/Labour/Operator → labour (COGS); others → staff (admin) | `config.labour_wage_table` / `staff_wage_table` | ① (blank → ② template rosters) |
| `priceEscalation` | Annual sale-price growth % | `config.annual_price_escalation_pct` | ① (default 5) |
| `electricityCost` | Monthly power bill → **× 12** for annual power cost | `config.electricity_monthly_inr` | ① (default 15,000) |
| `ownFunds` | FPO promoter equity committed | `inputs.own_equity_committed_lakh` | ① |
| `buyers`, `jobs` | Market & employment narrative | `narrative.*` | ① |
| `schemeChoice`, `schemeName` | Government scheme (or self-funded / "Others" → note) | `inputs.scheme.choice` | ① |
| `loan` `{interest_pct, tenure_months, moratorium_months}` | Loan terms; tenure ≤ 12 → default 60; moratorium clamped to tenure−1 | `config.interest_rate_pct` / `loan_tenor_months` / `moratorium_months` | ① (blank → ② template) |
| `docChecklist` | Evidence checklist — not consumed by the engine | — | ① |

---

## 4. Engine config parameters — what each controls & its source

These live in each template's `config` block. **Bold = the FPO can override it from the
form**; the rest are technical defaults the FPO never sees.

| Parameter | What it controls | Source & value |
|---|---|---|
| **`interest_rate_pct`** | Term-loan & WC interest rate | form `loan.interest_pct`, else template (11%) |
| **`loan_tenor_months`** | Loan repayment tenure | form `loan.tenure_months`, else template (60); forced ≥ 60 if ≤ 12 |
| **`moratorium_months`** | Interest-only grace period | form `loan.moratorium_months`, clamped ≤ tenure−1 |
| **`annual_price_escalation_pct`** | Yearly growth of sale price (and RM rate) | form `priceEscalation`, else **5** |
| `semi_variable_escalation_pct` | Yearly growth of labour/staff/power | template, **standardised to 5** |
| `rent_escalation_pct` | Yearly growth of rent (leased land) | template, **standardised to 10** |
| `capacity_utilisation_ramp_pct` | Year-by-year % of rated capacity used (5 values) | template (per line, e.g. 60→80%) |
| `depreciation_rate_pct` | WDV depreciation, plant vs misc | template (plant 10–15%, misc 10%) |
| `working_capital_cycle_days` | RM / finished-goods / receivable / payable days | template (per line) |
| `wc_margin_pct` | Promoter margin on working capital (rest = CC bank loan) | template (25%) |
| `target_dscr_min` | Reference DSCR — **label/reference only**, not enforced | template (1.5) |
| `repair_pct_of_sales`, `packaging_pct_of_sales`, `selling_admin_pct_of_sales` | Operating-cost ratios | template (per line) |
| `tax_rate_pct` | Corporate tax rate on PBT | template (25%, some lines 30.9%) |
| `tax_threshold_lakh` | **Removed** — tax is charged on the *full* PBT | template = 0 |
| `electricity_monthly_inr` | Monthly power bill; annual power = × 12 | form `electricityCost` (default 15,000) |
| `power` `{connection_kw, units_per_day, rate_per_unit}` | **Legacy** power formula — used only as a fallback when `electricity_monthly_inr` is absent | template |
| `labour_wage_table`, `staff_wage_table` | COGS labour vs admin staff rosters | form `manpower` (split), else template rosters |

---

## 5. Template-only technical norms (never from the form)

These are engineering assumptions the FPO can't reasonably supply, so they stay in the
template (`data/templates/<line>.json`), outside the `config` block:

| Field | Meaning |
|---|---|
| `capacity_input_unit`, `kg_per_capacity_unit`, `shift_hours` | Unit basis for converting the form's capacity to throughput |
| `technical_defaults.wastage_pct` | Process loss / yield (e.g. rice 33%, juice 0%) |
| `technical_defaults.pack_size_kg` | Packaging unit |
| `technical_defaults.misc_fixed_assets_lakh` | Default misc assets if the form leaves `otherFixedAssets` blank |
| `machinery_items_indicative` | Default machinery list if the form's `machinery[]` table is empty |
| `narrative.product_intro`, `raw_material_note` | Fixed-product intro (mills, ghee, etc.). For Oil/Juice/Spice these are `null` so the **LLM** writes a commodity-specific intro |
| `narrative.process_steps` | Drives the §4 process-flow chart |
| `revenue_model` | `processing` (RM → product) or `service` (CHC / warehouse: capacity is rented) |

---

## 6. What the engine computes (④ — derived, never an input)

Given ① ② ③, `financial_engine.compute()` produces `instance["computed"]`, which is
the source for every number in §1 and §3–§6:

production & sales · raw-material cost · fixed capital (machinery + civil + misc) ·
WDV depreciation · means of finance (subsidy / term loan / CC bank loan / FPO promoter
equity) · term-loan repayment schedule · working capital (margin + CC split) · P&L
(tax on full PBT) · break-even · balance sheet · cash flow · **DSCR** (annual + average).
The engine asserts **means of finance == total project cost** before returning.
Formulas: [docs/FINANCIAL_ENGINE.md](FINANCIAL_ENGINE.md).

---

## 7. What the LLM writes (⑤)

Only **§2 Product, Market & Value Chain** — Introduction, Uses & Benefits, Market
Potential, Value Chain, Competition, Risk & Mitigation, Employment, Sustainability. It
places `{{TABLE:id}}` markers where engine content goes and writes **no numbers**;
`render/assemble.py` injects the exact engine tables and `verify()` flags any figure in
the prose that doesn't match the engine.

---

## 8. Line defaults (③) — the per-line fallback set

Used when the form leaves capacity / price / raw-material blank. Capacity is modelled as
**output throughput in the line's own unit** (service lines use it as an annual figure).
Sources: the schema's `reference_data` for the 8 lines that carry defaults; our
calibrated base payloads for Dhal / Flour / Ghee / Spice.

| Line (template) | Unit | Default capacity | Working days | Sale price | Raw material |
|---|---|---|---|---|---|
| Rice Mill | kg/day | 375 | 300 | ₹47 | ₹22/unit |
| Dhal Mill | kg/day | 1,600 | 300 | ₹88 | ₹67.5/unit |
| Flour Mill | kg/day | 6,000 | 300 | ₹28 | ₹22/unit |
| Oil Extraction | Ltr/day | 237.5 | 300 | ₹140 | ₹35/unit |
| Ghee Processing | kg/day | 240 | 300 | ₹330 | ₹180/unit |
| Spice Processing | kg/day | 300 | 300 | ₹165 | ₹95/unit |
| Juice Processing | KL/day | 0.225 | 310 | ₹98,000 | 58% of revenue |
| Fruit & Veg Grading | kg/day | 575 | 300 | ₹16.2 | ₹9.8/unit |
| Nursery Hardening | plant/day | 145 | 310 | ₹16.5 | ₹6/unit |
| Vermicompost | MT/day | 0.29 | 310 | ₹4,880 | ₹132/unit |
| Custom Hiring Centre | machine-hr/yr | 5,000 *(annual)* | 305 | ₹680 | 30% of revenue |
| Storage / Warehouse | MT-year | 2,000 *(annual)* | 365 | ₹2,450 | 5% of revenue |

> ⚠️ **Note on defaults & viability:** some of these default *capacities* (e.g. Juice
> 0.225 KL/day, Rice 375 kg/day) are too small relative to the machinery/labour cost
> base, so an *accepted-defaults* submission can compute to a loss. The engine reports
> the loss honestly (it never fakes viability). For realistic, bankable defaults the
> `reference_data` capacity/price/machinery should be a mutually-consistent set —
> pending client confirmation.

---

## 9. Business line → template map

| Form `businessLine` | Template file |
|---|---|
| Rice Mill | `rice_mill` |
| Dhal Mill | `dal_mill` |
| Flour Mill | `flour_mill` |
| Oil Extraction | `oil_extraction_unit` |
| Ghee Processing | `processing_unit_ghee` |
| Spice Processing | `processing_unit_spices` |
| Juice Processing | `processing_unit_fruit_juice` |
| Fruit & Vegetable Grading | `veg_fruit_grading_sorting_unit` |
| Nursery Hardening | `nursery_tissue_culture_hardening_unit` |
| Vermicompost | `vermicompost_organic_fertilizer_unit` |
| Custom Hiring Centre | `custom_hiring_centre` |
| Storage / Warehouse | `warehouse_cold_storage` |
