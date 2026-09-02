"""
DPR section structure (Deliverable 4).

Renders the engine's `computed` tables into the exact NIFTEM ghee model-DPR
section order, plus the mandatory "Limitations of the Model DPR" section.

Two-lane note: this module only *formats* numbers the engine already computed —
it performs no arithmetic. In production the LLM does the narrative prose
(About the Product, market text, etc.); this file gives the deterministic
skeleton + all financial tables so the structure and figures are guaranteed.
"""
from __future__ import annotations

YRS = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]

# The exact section list (matches the NIFTEM ghee model DPR).
SECTION_ORDER = [
    "1. Project Summary",
    "2. Product, Market & Value Chain",
    "3. Basis & Presumptions",
    "4. Process Flow Chart",
    "5. Economics of the Project",
    "   5.1 Key Operating Parameters",
    "   5.2 Capacity, Utilisation, Production & Output",
    "   5.3 Premises",
    "   5.4 Machinery & Equipment",
    "   5.5 Misc Fixed Assets",
    "   5.6 Total Cost of Project",
    "   5.7 Means of Finance",
    "   5.8 Term Loan",
    "   5.9 Repayment & Interest Schedule",
    "   5.10 Working Capital",
    "   5.11 Salaries & Wages",
    "   5.12 Power Requirement",
    "   5.13 Depreciation",
    "   5.14 Repair & Maintenance",
    "   5.15 Profitability Analysis",
    "   5.16 Break-Even Point",
    "   5.17 Projected Balance Sheet",
    "   5.18 Cash-Flow Statement",
    "   5.19 Debt-Service Coverage Ratio",
    "6. Limitations of the Model DPR",
]

LIMITATIONS_TEXT = """## 6. Limitations of the Model DPR

i. This DPR provides the basic standard components and methodology for a proposal; it is not a substitute for the entrepreneur's own due diligence.

ii. Entrepreneur-, crop-, and location-specific details (business constitution, project location, raw-material sourcing, SWOT, detailed market research, project rationale for the specific location, community benefit, and employment generation) must be added by the FPO.

iii. The projections rest on assumptions about cost, prices, interest, capacity utilisation and output recovery that vary by place, market and situation; actual figures will differ and the calculations change accordingly.

iv. The means of finance shown (subsidy / promoter contribution / term loan) follows the selected scheme's structure; a different scheme or a credit-linked variant changes the numbers without changing the methodology.

The entrepreneur must verify machinery quotes by demonstration, ensure reliable raw-material supply, maintain optimum inventory, keep a sound debt-equity ratio, and provide all documents required by the sanctioning authority."""


DISCLAIMER_TEXT = """## Disclaimer

This document has been generated with the assistance of an Artificial Intelligence (AI) system developed by Avataar AI in collaboration with the NAFPO, based on data and information provided by the concerned Farmer Producer Organisation (FPO), publicly available government scheme guidelines, and other secondary sources.

This is a computer-generated document. While reasonable efforts have been made to align its content, financial projections, and assumptions with prevailing government scheme norms and market data available at the time of generation, the AI system may not have access to the most current, localised, or verified information, and its outputs may contain errors, omissions, or assumptions that do not fully reflect ground realities.

This document is intended to serve as a preliminary draft / starting point for business planning and is not a substitute for professional due diligence. All financial figures, cost estimates, technical specifications, market assumptions, and regulatory references contained herein must be independently verified and validated by the concerned FPO, its Cluster Based Business Organisation (CBBO) / promoting institution, a qualified financial or technical consultant, and the appropriate bank / financial institution before being relied upon for investment, loan appraisal, or submission to any government authority.

The views, projections, and conclusions presented, including all financial estimates, do not represent or reflect the policy, endorsement, or assurance of NAFPO, Avataar AI, or any government department, bank, or financial institution. NAFPO and Avataar AI accept no liability, financial or otherwise, and make no warranty — express or implied — as to the accuracy, completeness, or fitness for purpose of this document. Any decision made on the basis of this document is at the sole discretion and risk of the user."""


def _tbl(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _series(label: str, arr: list) -> list:
    return [label] + [f"{x:,.2f}" for x in arr]


def render(instance: dict, narrative_md: str | None = None) -> str:
    """narrative_md: the LLM-written narrative block for section 2 (product,
    market, value chain, risk, etc., using its own ### sub-headings). If None,
    falls back to the narrative fields in the instance (or a placeholder)."""
    c = instance["computed"]
    n = instance.get("narrative", {})
    inp = instance["inputs"]
    meta = instance.get("meta", {})
    # Service lines (CHC / warehouse) rent capacity — no raw-material -> product
    # transformation, so several tables are relabelled and wastage/production drop.
    is_service = c.get("meta", {}).get("revenue_model") == "service"
    # Batch/annual-throughput processing (nursery/hardening): capacity is annual
    # input units and "wastage" is plant mortality.
    is_batch = (not is_service) and ("annual_input_units" in inp.get("capacity", {}))
    # Physical quantity unit for production/RM rows ("kg" default, "plant" for a
    # nursery, "litre" for oil, ...). Service lines don't use it.
    unit = "units" if is_service else (meta.get("product_unit") or "kg")
    loss_word = "mortality" if is_batch else "wastage"
    # tidy quantity: no ".00" on whole numbers, 2dp otherwise (e.g. 1.5 KL, 2,000 kg)
    q = lambda x: f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"
    P: list[str] = []

    # --- Title ---
    P.append(f"# {meta.get('report_title', 'DETAILED PROJECT REPORT')}")
    if meta.get("synthetic"):
        P.append("*Synthetic model DPR generated by the FPO DPR pipeline. "
                 "All financial figures are engine-computed.*")

    # --- 1. Project Summary ---
    mf, pc = c["means_of_finance"], c["project_cost"]
    prof, dscr = c["profitability"], c["dscr"]
    _prod = meta.get("commodity", "").title() or "*To be furnished*"
    _focus = n.get("commodity_focus")
    prod_rows = [["Product", _prod]]
    # show the FPO's selected crop/commodity when it isn't already the product name
    # (e.g. Nursery / Grading / Storage where the product is generic)
    if _focus and _focus.lower() not in _prod.lower():
        prod_rows.append(["Commodity / Focus", str(_focus).title()])
    summary_rows = [
        ["Name of the FPO", n.get("fpo_name") or "*To be furnished*"],
        ["Location", n.get("location") or "*To be furnished*"],
        ["CIN / Registration No.", n.get("registration_no") or "*To be furnished*"],
        ["CEO / Director", n.get("ceo_name") or "*To be furnished*"],
        ["Mobile", n.get("mobile") or "*To be furnished*"],
        ["E-mail", n.get("email") or "*To be furnished*"],
        ["Address of Proposed Site", n.get("site_address") or "*To be furnished*"],
        *prod_rows,
        ["Land status", inp["land_status"]],
        ["Scheme", mf["scheme"]],
        ["Total Project Cost (Rs. Lakh)", f"{pc['total_project_cost']:,.2f}"],
        ["Means of Finance", mf["funding_basis"]],
    ]
    P.append("## 1. Project Summary\n\n" + _tbl(["Item", "Detail"], summary_rows))

    # --- 2. Product, Market & Value Chain (narrative lane; the LLM writes and
    #        organises this whole block with its own ### sub-headings) ---
    if narrative_md:
        P.append("## 2. Product, Market & Value Chain\n\n" + narrative_md.strip())
    else:
        about = []
        if n.get("product_intro"):
            about.append(f"### Introduction\n\n{n['product_intro']}")
        if n.get("product_uses_benefits"):
            about.append(f"### Uses & Benefits\n\n{n['product_uses_benefits']}")
        if n.get("market_potential"):
            about.append(f"### Market Potential\n\n{n['market_potential']}")
        if n.get("raw_material_note"):
            about.append(f"### Raw Material\n\n{n['raw_material_note']}")
        P.append("## 2. Product, Market & Value Chain\n\n" + ("\n\n".join(about) if about
                 else "*[LLM narrative block: product, uses/benefits, market potential, value chain, risk.]*"))

    # --- 3. Basis & Presumptions (deterministic, from config) ---
    cfg = instance["config"]
    cap_in = inp["capacity"]
    d = cfg["working_capital_cycle_days"]
    dep = cfg["depreciation_rate_pct"]
    ramp = cfg["capacity_utilisation_ramp_pct"]
    rm_method = inp["prices"]["rm_costing_method"]
    rm_val = inp["prices"]["raw_material_price"]
    rm_desc = (f"{rm_val*100:.0f}% of revenue" if rm_method == "pct_of_revenue"
               else f"Rs. {rm_val:,.0f} per unit")
    if is_service:
        cap_line = (
            f"1. Rated annual capacity is {c['capacity_production']['rm_requirement_100_kg']:,.0f} "
            f"service units over {cap_in['working_days_per_annum']} operating days per annum; "
            f"first year is taken at {ramp[0]}% capacity utilisation.")
        conv_line = ("3. There is no raw-material-to-product conversion; the rented capacity is "
                     "billed directly, so no processing wastage applies.")
        price_line = (f"4. Rental / hire charge is Rs. {inp['prices']['sale_price']:,.0f} per service unit; "
                      f"direct operating cost (fuel, consumables, upkeep) is taken as {rm_desc}.")
    elif is_batch:
        cap_line = (
            f"1. Rated annual capacity is {c['capacity_production']['rm_requirement_100_kg']:,.0f} "
            f"{unit}s of input over {cap_in['working_days_per_annum']} operating days per annum "
            f"(across hardening cycles); first year is taken at {ramp[0]}% capacity utilisation.")
        conv_line = (f"3. Plant mortality of {cap_in['wastage_pct']}% is considered to arrive at "
                     f"the net saleable output.")
        price_line = (f"4. Input (plantlet) cost is taken as {rm_desc}; sale price is "
                      f"Rs. {inp['prices']['sale_price']:,.0f} per {unit}.")
    else:
        _per_day = cap_in['per_hr_kg'] * cap_in['shift_hours']
        cap_line = (
            f"1. Production capacity is {q(_per_day)} {unit}/day "
            f"({q(cap_in['per_hr_kg'])} {unit}/hr over a {cap_in['shift_hours']}-hour shift) "
            f"for {cap_in['working_days_per_annum']} working days per annum; "
            f"first year is taken at {ramp[0]}% capacity utilisation.")
        conv_line = (f"3. Wastage of {cap_in['wastage_pct']}% is considered on raw material to arrive at net output."
                     if cap_in['wastage_pct'] else
                     "3. The stated capacity is taken as the net saleable output; costs are computed on this "
                     "throughput (no separate processing-loss adjustment).")
        price_line = (f"4. Raw-material cost is taken as {rm_desc}; sale price is "
                      f"Rs. {inp['prices']['sale_price']:,.0f} per unit.")
    basis = [
        cap_line,
        f"2. Capacity utilisation is assumed at {', '.join(str(r)+'%' for r in ramp)} in years 1–5 respectively.",
        conv_line,
        price_line,
        f"5. Sales and raw-material rates are escalated at {cfg['annual_price_escalation_pct']}% per year; "
        f"labour, staff and power costs at {cfg.get('semi_variable_escalation_pct', 0)}% per year.",
        f"6. Raw-material stock is held for {d.get('raw_material_days')} days and finished goods for "
        f"{d.get('finished_goods_days')} days; debtors are given {d.get('receivables_days')} days and "
        f"creditors allow {d.get('payables_days')} days.",
        f"7. Working-capital margin is {cfg.get('wc_margin_pct')}% (the balance is bank finance).",
        f"8. Depreciation is on the WDV method — plant & machinery at {dep['plant_machinery']}% and "
        f"miscellaneous assets at {dep['misc_assets']}%.",
        f"9. Interest on term loan and working capital is taken at {cfg['interest_rate_pct']}% p.a.; the term "
        f"loan runs {cfg['loan_tenor_months']} months with a {cfg['moratorium_months']}-month moratorium.",
        f"10. Income tax is provided at {cfg.get('tax_rate_pct', 0)}% "
        + (f"on profit above Rs. {cfg.get('tax_threshold_lakh', 0):.2f} lakh. " if cfg.get('tax_threshold_lakh') else "of profit before tax. ")
        + f"Salary and wage rates are as per current market. Premises are {inp['land_status'].replace('_', ' ')}.",
        f"11. Means of finance: {c['means_of_finance']['funding_basis']}",
    ]
    P.append("## 3. Basis & Presumptions\n\n" + "\n".join(basis))

    # --- 4. Process Flow Chart ---
    steps = n.get("process_steps", []) or []
    flow = "\n".join(f"{i}. {s}" + ("  \n   |\n   v" if i < len(steps) else "")
                     for i, s in enumerate(steps, 1)) if steps else "*To be furnished*"
    P.append("## 4. Process Flow Chart\n\n" + flow)

    # --- 5. Economics of the Project ---
    P.append("## 5. Economics of the Project")

    cap = c["capacity_production"]
    if is_service:
        op_rows = [
            ["Rated annual capacity (service units/yr)", f"{cap['rm_requirement_100_kg']:,.0f}"],
            ["Operating days / annum", inp["capacity"]["working_days_per_annum"]],
            ["Rental / hire charge (Rs./unit)", f"{inp['prices']['sale_price']:,.0f}"],
            ["Direct operating cost", f"{inp['prices']['raw_material_price']*100:.0f}% of revenue"],
            ["Interest rate (%)", instance["config"]["interest_rate_pct"]],
            ["Price escalation (%/yr)", instance["config"]["annual_price_escalation_pct"]]]
    elif is_batch:
        op_rows = [
            [f"Annual input capacity ({unit}s/yr)", f"{cap['rm_requirement_100_kg']:,.0f}"],
            ["Operating days / annum", inp["capacity"]["working_days_per_annum"]],
            ["Plant mortality (%)", inp["capacity"]["wastage_pct"]],
            [f"Plantlets required ({unit}s/yr)", f"{cap['rm_requirement_100_kg']:,.0f}"],
            [f"Saleable output after mortality ({unit}s/yr)", f"{cap['output_100_kg']:,.0f}"],
            ["Interest rate (%)", instance["config"]["interest_rate_pct"]],
            ["Price escalation (%/yr)", instance["config"]["annual_price_escalation_pct"]]]
    else:
        _per_day = inp["capacity"]["per_hr_kg"] * inp["capacity"]["shift_hours"]
        _w = inp["capacity"]["wastage_pct"]
        op_rows = [
            [f"Rated capacity ({unit}/day)", q(_per_day)],
            [f"Rated capacity ({unit}/hr)", q(inp["capacity"]["per_hr_kg"])],
            ["Shift hours", inp["capacity"]["shift_hours"]],
            ["Working days / annum", inp["capacity"]["working_days_per_annum"]]]
        if _w:                                    # only show wastage when it actually applies
            op_rows += [
                ["Wastage (%)", _w],
                [f"Raw material requirement ({unit}/yr)", f"{cap['rm_requirement_100_kg']:,.0f}"],
                [f"Output after wastage ({unit}/yr)", f"{cap['output_100_kg']:,.0f}"]]
        else:                                     # input == output; a single throughput figure
            op_rows += [[f"Annual throughput ({unit}/yr)", f"{cap['output_100_kg']:,.0f}"]]
        op_rows += [
            ["Interest rate (%)", instance["config"]["interest_rate_pct"]],
            ["Price escalation (%/yr)", instance["config"]["annual_price_escalation_pct"]]]
    P.append("### 5.1 Key Operating Parameters\n\n" + _tbl(["Parameter", "Value"], op_rows))

    if is_service:
        util_rows = [
            ["Capacity Utilisation (%)"] + instance["config"]["capacity_utilisation_ramp_pct"],
            ["Service Units Rented"] + [f"{x:,.0f}" for x in cap["production_kg"]],
            _series("Revenue (Rs. Lakh)", cap["sales_lakh"]),
            _series("Direct Operating Cost (Rs. Lakh)", cap["rm_cost_lakh"])]
    else:
        util_rows = [
            ["Capacity Utilisation (%)"] + instance["config"]["capacity_utilisation_ramp_pct"],
            [f"Production ({unit + 's' if unit.isalpha() and unit.islower() and len(unit) > 3 else unit})"] + [f"{x:,.0f}" for x in cap["production_kg"]],
            _series("Sales (Rs. Lakh)", cap["sales_lakh"]),
            _series("Raw Material Cost (Rs. Lakh)", cap["rm_cost_lakh"])]
    P.append("### 5.2 Capacity, Utilisation, Production & Output\n\n"
             + _tbl(["Particulars"] + YRS, util_rows))

    _ls = str(inp["land_status"])
    prem = {"owned": "Owned premises (no rent)", "leased": "Leased premises (rent in P&L)",
            "to_be_acquired": "To be acquired (civil in capex)"}.get(
        _ls, "Leased premises (rent in P&L)" if ("leas" in _ls or "rent" in _ls)
        else "Owned premises (no rent)")   # tolerate any land_status label
    P.append(f"### 5.3 Premises\n\n{prem}.")

    items = inp["capex"].get("machinery_items", [])
    # Items are normally {"name","inr"} dicts, but a payload may list bare names with
    # no per-item cost; apportion the (authoritative) machinery total evenly over those
    # so the breakdown still sums to the Total Machinery row.
    n = len(items) or 1
    def _mrow(m):
        if isinstance(m, dict):
            return [m.get("name", "Machinery"), f"{m.get('inr', 0) / 1e5:,.2f}"]
        return [str(m), f"{pc['machinery'] / n:,.2f}"]
    mrows = [_mrow(m) for m in items]
    mrows.append(["**Total Machinery**", f"**{pc['machinery']:,.2f}**"])
    P.append("### 5.4 Machinery & Equipment (Rs. Lakh)\n\n" + _tbl(["Item", "Amount"], mrows))

    P.append("### 5.5 Misc Fixed Assets (Rs. Lakh)\n\n" + _tbl(
        ["Item", "Amount"], [["Misc. fixed assets", f"{pc['misc_fixed_assets']:,.2f}"]]))

    P.append("### 5.6 Total Cost of Project (Rs. Lakh)\n\n" + _tbl(
        ["Component", "Amount"],
        [["Land & Building", "Owned/Rented"],
         ["Plant & Machinery", f"{pc['machinery']:,.2f}"],
         ["Miscellaneous Assets", f"{pc['misc_fixed_assets']:,.2f}"],
         ["Civil Construction", f"{pc['civil']:,.2f}"],
         ["Working Capital", f"{pc['working_capital_gross_y1']:,.2f}"],
         ["**Total Project Cost**", f"**{pc['total_project_cost']:,.2f}**"]]))

    mf_rows = [["FPO Promoter's Contribution", f"{mf['own_equity']:,.2f}"]]
    if mf["subsidy"] > 0:
        mf_rows.append(["Subsidy", f"{mf['subsidy']:,.2f}"])
    mf_rows += [["Term Loan", f"{mf['term_loan']:,.2f}"],
                ["Working-Capital Bank Loan", f"{mf['wc_bank_loan']:,.2f}"],
                ["**Total**", f"**{mf['total']:,.2f}**"]]
    P.append(f"### 5.7 Means of Finance (Rs. Lakh)\n\n_{mf['funding_basis']}_\n\n"
             + _tbl(["Source", "Amount"], mf_rows)
             + f"\n\nMinimum FPO promoter contribution required: Rs. {mf['min_contribution_required']:,.2f} Lakh — "
             + ("**met**." if mf["min_contribution_ok"] else "**not met (review).**")
             + (f"\n\n_Note: {mf['fallback_note']}_" if mf.get("fallback_note") else ""))

    P.append(f"### 5.8 Term Loan\n\nTerm loan of **Rs. {mf['term_loan']:,.2f} Lakh** at "
             f"{instance['config']['interest_rate_pct']}% p.a., tenor "
             f"{instance['config']['loan_tenor_months']} months, moratorium "
             f"{instance['config']['moratorium_months']} months.")

    tl = c["term_loan_schedule"]
    P.append("### 5.9 Repayment & Interest Schedule (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Principal Repaid", tl["principal"]),
         _series("Interest", tl["interest"]),
         _series("Closing Balance", tl["closing_balance"])]))

    wc = c["working_capital"]
    P.append("### 5.10 Working Capital (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Gross Working Capital", wc["gross"]),
         _series("Margin", wc["margin"]),
         _series("Bank Finance", wc["bank"])]))

    cfg = instance["config"]
    labour_rows = [[w["role"], w["count"], f"{w['monthly_wage_inr']:,}"]
                   for w in cfg.get("labour_wage_table", [])]
    staff_rows = [[w["role"], w["count"], f"{w['monthly_wage_inr']:,}"]
                  for w in cfg.get("staff_wage_table", [])]
    P.append("### 5.11 Salaries & Wages\n\n"
             + "**Labour (in cost of production):**\n\n"
             + _tbl(["Role", "Count", "Monthly Wage (Rs.)"], labour_rows)
             + "\n\n**Staff salary (administrative):**\n\n"
             + _tbl(["Role", "Count", "Monthly Wage (Rs.)"], staff_rows)
             + "\n\n"
             + _tbl(["Particulars"] + YRS,
                    [_series("Wages & Labour (Rs. Lakh)", c["profitability"]["wages_labour"]),
                     _series("Salary to Staff (Rs. Lakh)", c["profitability"]["salary_staff"])]))

    P.append("### 5.12 Power Requirement (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS, [_series("Power Cost", prof["electricity"])]))

    dep = c["depreciation"]
    P.append("### 5.13 Depreciation (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Depreciation", dep["annual"]), _series("Net Block", dep["net_block"])]))

    P.append("### 5.14 Repair & Maintenance (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS, [_series("Repairs & Maintenance (3% of sales)", prof["repair"])]))

    P.append("### 5.15 Projections of Profitability Analysis (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Gross Sale", prof["sales"]),
         _series("Direct Operating Cost" if is_service else "Raw Material Consumed", prof["rm_cost"]),
         _series("Electricity", prof["electricity"]),
         _series("Depreciation", prof["depreciation"]),
         _series("Wages & Labour", prof["wages_labour"]),
         _series("Repair & Maintenance", prof["repair"]),
         _series("Packaging", prof["packaging"]),
         _series("Cost of Sales", prof["cost_of_sales"]),
         _series("**Gross Profit**", prof["gross_profit"]),
         _series("Salary to Staff", prof["salary_staff"]),
         _series("Interest on Term Loan", prof["interest_term_loan"]),
         _series("Interest on Working Capital", prof["interest_wc"]),
         _series("Rent", prof["rent"]),
         _series("Selling & Admin Expenses", prof["selling_admin"]),
         _series("Profit Before Tax", prof["pbt"]),
         _series("Taxation", prof["tax"]),
         _series("**Profit After Tax**", prof["pat"])]))

    be = c["break_even"]
    P.append("### 5.16 Break-Even Point (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Variable Cost", be["variable_cost"]),
         _series("Fixed Cost", be["fixed_cost"]),
         _series("Contribution", be["contribution"]),
         ["Break-Even (%)"] + [f"{x:,.2f}" for x in be["bep_pct"]]]))

    bs = c["balance_sheet"]
    P.append("### 5.17 Projected Balance Sheet (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Net Worth", bs["net_worth"]),
         _series("Term Loan (closing)", bs["term_loan_closing"]),
         _series("Net Fixed Assets", bs["net_block"]),
         _series("Cash & Bank", bs["closing_cash"])]))

    cf = c["cash_flow"]
    P.append("### 5.18 Cash-Flow Statement (Rs. Lakh)\n\n" + _tbl(
        ["Particulars"] + YRS,
        [_series("Cash Accrual", cf["cash_accrual"]),
         _series("Loan Repayment", cf["loan_repayment"]),
         _series("Closing Cash", cf["closing_cash"])]))

    P.append("### 5.19 Debt-Service Coverage Ratio\n\n" + _tbl(
        ["Particulars"] + YRS, [["DSCR"] + [f"{x:,.2f}" for x in dscr["annual"]]])
        + f"\n\n**Average DSCR: {dscr['average']:,.2f}**")

    # --- 5. Limitations ---
    P.append(LIMITATIONS_TEXT)

    return "\n\n".join(P).strip() + "\n"


# Small helpers to recompute two display-only series from config (deterministic,
# mirror the engine's formulas so the tables show power & repairs by year).
def _power_series(instance):
    cfg, inp = instance["config"], instance["inputs"]
    ramp = [p / 100 for p in cfg["capacity_utilisation_ramp_pct"]]
    esc = cfg["annual_price_escalation_pct"] / 100
    pw = cfg.get("power", {})
    kwh = pw.get("connected_load_kw", 0) * pw.get("load_factor", 1) * \
        inp["capacity"]["shift_hours"] * inp["capacity"]["working_days_per_annum"]
    full = kwh * pw.get("rate_per_kwh", 0) / 1e5
    return [full * ramp[y] * (1 + esc) ** y for y in range(5)]


def _repairs_series(instance):
    cfg, inp = instance["config"], instance["inputs"]
    esc = cfg["annual_price_escalation_pct"] / 100
    mach = inp["capex"]["machinery_inr"] / 1e5
    return [mach * cfg.get("repairs_pct_of_machinery", 0) / 100 * (1 + esc) ** y for y in range(5)]


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    from dpr import config
    from dpr.engine.financial_engine import compute
    inst = json.load(open(config.GHEE_FIXTURE))
    print(render(compute(inst)))
