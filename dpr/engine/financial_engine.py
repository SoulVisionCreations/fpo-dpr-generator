"""
Deterministic financial engine for FPO DPRs (base commodity: ghee).

Calibrated to the NIFTEM PM-FME "Ghee Making Unit" model DPR. Reproduces its
basis & presumptions exactly:
  * RM cost on the raw-material REQUIREMENT (gross throughput, wastage NOT
    deducted); output/sales on net-of-wastage production less 5-day FG stock.
  * No preliminary/pre-operative expense. TPC = fixed capital + gross WC.
  * PM-FME means of finance = 35% subsidy / 55% term loan / 10% promoter of
    fixed capital; WC financed 90% bank / 10% margin (traditional method).
  * P&L keeps labour (in COGS) and staff salary (below gross profit) separate,
    repair = 3% of sales, rent escalated, packaging & selling as % of sales.

Two-lane rule: all arithmetic is here; the LLM only renders prose.
All money in Rs. Lakh; INR inputs converted with /1e5.

Branches handled: scheme (no_scheme|pmfme|stubs), land (owned|leased|
to_be_acquired), rm costing (per_unit|pct_of_revenue).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dpr.engine.scheme_norms import SCHEME_NORMS
from dpr import config

LAKH = 1e5
YEARS = 5


def _r(x, n=2):
    return round(float(x), n)


class DPRFinancialModel:
    def __init__(self, instance: dict):
        self.inst = instance
        self.inp = instance["inputs"]
        self.cfg = instance["config"]
        self.days = self.inp["capacity"]["working_days_per_annum"]
        self.ramp = [p / 100.0 for p in self.cfg["capacity_utilisation_ramp_pct"]]
        self.esc = self.cfg["annual_price_escalation_pct"] / 100.0
        # Semi-variable costs (labour/staff/power) escalate slower than prices.
        self.esc_semi = self.cfg.get("semi_variable_escalation_pct", 0) / 100.0
        # "processing" (RM -> product) or "service" (CHC / warehouse: capacity is
        # rented, no raw-material -> product transformation).
        self.revenue_model = instance.get("meta", {}).get("revenue_model", "processing")

    def compute(self) -> dict:
        self._production_sales()
        self._fixed_capital()
        self._depreciation()
        self._operating_costs()     # needs sales & dep
        self._term_loan_schedule()
        self._working_capital()     # needs COP -> so after operating costs
        self._project_cost_and_finance()
        self._pnl()
        self._break_even()
        self._balance_sheet()
        self._cash_flow()
        self._dscr()
        self._assemble_computed()
        self._assert_balance()
        return self.inst

    # --- semi-variable escalator (labour/staff/power) ---------------------
    # Default (ghee): the full roster runs from year 1, so costs are scaled
    #   relative to the year-1 utilisation and grow with the ramp.
    # Opt-in via config `labour_scales_with_utilisation: true` (e.g. Amla):
    #   costs scale by ABSOLUTE utilisation, so a low-capacity year 1 uses
    #   proportionally less labour/power. Both then escalate at esc_semi.
    def _semi(self, base, y):
        factor = self.ramp[y] if self.cfg.get("labour_scales_with_utilisation", False) \
            else self.ramp[y] / self.ramp[0]
        return base * factor * (1 + self.esc_semi) ** y

    # --- capacity / production / sales / raw material (branch C) --------- #
    def _production_sales(self):
        cap = self.inp["capacity"]
        pr = self.inp["prices"]
        self.rm_method = pr["rm_costing_method"]
        # Year 1 uses the exact base price; escalated years are rupee-rounded
        # (half-up) to mimic the model DPRs. Rounding Y1 would corrupt genuine
        # paise prices (e.g. dal RM Rs 67.5 -> 68).
        rup = lambda x: int(x + 0.5)
        self.sale_price = [pr["sale_price"] if y == 0 else rup(pr["sale_price"] * (1 + self.esc) ** y)
                           for y in range(YEARS)]
        self.rm_rate = [pr["raw_material_price"] if y == 0 else rup(pr["raw_material_price"] * (1 + self.esc) ** y)
                        for y in range(YEARS)]

        if self.revenue_model == "service":
            # Capacity is rented service units (machine-hours, MT-storage, ...);
            # utilisation is the ramp. No wastage, no finished-goods stock.
            rated = cap["annual_service_units"]
            self.rm_requirement_100 = self.output_100 = rated
            self.rm_qty = [rated * self.ramp[y] for y in range(YEARS)]
            self.production = [rated * self.ramp[y] for y in range(YEARS)]
            self.fg_open_kg = [0.0] * YEARS
            self.fg_close_kg = [0.0] * YEARS
            self.net_sale_kg = list(self.production)
        else:
            # Batch/annual-throughput processing (e.g. nursery/hardening unit):
            # capacity is given directly as annual input units (plants/yr =
            # plants-per-cycle x cycles), and "wastage" is plant mortality.
            # Otherwise the classic per-hour x shift x days throughput.
            if "annual_input_units" in cap:
                rm_req = cap["annual_input_units"]
            else:
                rm_req = cap["per_hr_kg"] * cap["shift_hours"] * cap["working_days_per_annum"]
            wast = cap["wastage_pct"] / 100.0
            self.rm_requirement_100 = rm_req
            self.output_100 = rm_req * (1 - wast)
            self.rm_qty = [rm_req * self.ramp[y] for y in range(YEARS)]
            self.production = [self.output_100 * self.ramp[y] for y in range(YEARS)]
            fg_days = self.cfg["working_capital_cycle_days"]["finished_goods_days"]
            self.fg_close_kg, self.fg_open_kg, self.net_sale_kg = [], [], []
            prev_close = 0.0
            for y in range(YEARS):
                close = self.production[y] / self.days * fg_days
                self.fg_open_kg.append(prev_close)
                self.fg_close_kg.append(close)
                self.net_sale_kg.append(self.production[y] + prev_close - close)
                prev_close = close

        self.sales = [self.net_sale_kg[y] * self.sale_price[y] / LAKH for y in range(YEARS)]
        if self.rm_method == "per_unit":
            self.rm_cost = [self.rm_qty[y] * self.rm_rate[y] / LAKH for y in range(YEARS)]
        elif self.rm_method == "pct_of_revenue":
            frac = pr["raw_material_price"]
            self.rm_cost = [self.sales[y] * frac for y in range(YEARS)]
        else:
            raise ValueError(f"unknown rm_costing_method: {self.rm_method}")

    # --- fixed capital (branch B: land) --------------------------------- #
    def _fixed_capital(self):
        capex, land = self.inp["capex"], self.inp["land_status"]
        self.machinery = capex["machinery_inr"] / LAKH
        self.misc_assets = capex["misc_fixed_assets_lakh"]
        # Civil construction is ALWAYS capitalised whenever a cost is reported,
        # regardless of land status (owned / leased / to-be-acquired) — a shed or
        # civil works built by the FPO is its asset even on leased land. Leased land
        # additionally carries a monthly rent, charged to the P&L (not capitalised).
        self.civil = capex.get("civil_construction_lakh") or 0.0
        self.rent_base = (self.inp.get("rent_per_month_inr", 0) or 0) * 12 / LAKH \
            if land == "leased" else 0.0
        # No preliminary/pre-operative expense (matches the model DPR).
        self.fixed_capital = self.machinery + self.misc_assets + self.civil

    def _depreciation(self):
        dp = self.cfg["depreciation_rate_pct"]
        pr, mr = dp["plant_machinery"] / 100.0, dp["misc_assets"] / 100.0
        self.dep, self.net_block = [], []
        plant, misc = self.machinery + self.civil, self.misc_assets
        for _ in range(YEARS):
            d = plant * pr + misc * mr
            self.dep.append(d)
            plant *= (1 - pr); misc *= (1 - mr)
            self.net_block.append(plant + misc)

    # --- operating cost lines ------------------------------------------- #
    def _operating_costs(self):
        cfg = self.cfg
        labour_base = sum(w["count"] * w["monthly_wage_inr"] for w in cfg["labour_wage_table"]) * 12 / LAKH
        staff_base = sum(w["count"] * w["monthly_wage_inr"] for w in cfg["staff_wage_table"]) * 12 / LAKH
        # Power cost: prefer the form's electricity bill (Rs/month x 12); fall back
        # to the legacy connected-load formula (units/day x days x rate) when the
        # instance carries no monthly figure (older instances / template default).
        em = cfg.get("electricity_monthly_inr")
        if em is not None:
            power_base = float(em) * 12 / LAKH
        else:
            pw = cfg.get("power", {})
            power_base = pw.get("units_per_day", 0) * self.days * pw.get("rate_per_unit", 0) / LAKH

        self.labour = [self._semi(labour_base, y) for y in range(YEARS)]
        self.staff = [self._semi(staff_base, y) for y in range(YEARS)]
        self.electricity = [self._semi(power_base, y) for y in range(YEARS)]
        self.repair = [self.sales[y] * cfg.get("repair_pct_of_sales", 0) / 100.0 for y in range(YEARS)]
        self.packaging = [self.sales[y] * cfg.get("packaging_pct_of_sales", 0) / 100.0 for y in range(YEARS)]
        self.selling_adm = [self.sales[y] * cfg.get("selling_admin_pct_of_sales", 0) / 100.0 for y in range(YEARS)]
        re = cfg.get("rent_escalation_pct", 0) / 100.0
        self.rent = [self.rent_base * (1 + re) ** y for y in range(YEARS)]

        # Cost of production & finished-goods stock value (for cost of sales / WC).
        self.cop = [self.rm_cost[y] + self.electricity[y] + self.dep[y] + self.labour[y]
                    + self.repair[y] + self.packaging[y] for y in range(YEARS)]
        fg_days = cfg["working_capital_cycle_days"]["finished_goods_days"]
        self.fg_close_val, self.fg_open_val = [], []
        prev = 0.0
        for y in range(YEARS):
            val = self.cop[y] / self.days * fg_days
            self.fg_open_val.append(prev); self.fg_close_val.append(val); prev = val

    def _term_loan_schedule(self):
        # term_loan amount is set in _project_cost_and_finance, but schedule needs
        # it first -> compute the loan amount here from fixed capital & scheme.
        self._means_of_finance_amounts()
        amt = self.term_loan
        # effective_interest_rate is set per scheme (AIF lowers it via subvention).
        rate = self.effective_interest_rate / 100.0
        tenor, morat = self.cfg["loan_tenor_months"], self.cfg["moratorium_months"]
        principal_pm = amt / (tenor - morat) if tenor > morat else amt / tenor
        self.tl_principal = [0.0] * YEARS
        self.tl_interest = [0.0] * YEARS
        self.tl_closing = [0.0] * YEARS
        bal, month = amt, 0
        for y in range(YEARS):
            for _ in range(12):
                month += 1
                interest = 0.0 if month == 1 else bal * rate / 12.0  # disbursed end of month 1
                principal = 0.0 if month <= morat else min(principal_pm, bal)
                bal = max(bal - principal, 0.0)
                self.tl_principal[y] += principal
                self.tl_interest[y] += interest
            self.tl_closing[y] = bal

    def _means_of_finance_amounts(self):
        choice = self.inp["scheme"]["choice"]
        fc = self.fixed_capital
        self.eligible_fc = self.machinery + self.misc_assets + self.civil
        base_rate = self.cfg["interest_rate_pct"]
        self.effective_interest_rate = base_rate          # default; AIF lowers it
        # Scheme fallback: an unknown, unmodelled or stub scheme (e.g. the form's
        # "Others" / "Not sure", or a stub with no financing type) falls back to a
        # self-funded structure, and a note is surfaced into the DPR so the reader
        # knows the named scheme wasn't modelled.
        self.scheme_fallback_note = None
        norm = SCHEME_NORMS.get(choice)
        if norm is None or norm.get("type") is None:
            self.scheme_fallback_note = (
                f"The selected financing option ('{choice}') is not modelled in this tool; "
                "the project is presented on a self-funded basis (FPO promoter equity + term "
                "loan, no capital subsidy). Please consult your CBBO / bank for the specific "
                "terms of the intended scheme.")
            norm = SCHEME_NORMS["no_scheme"]
        stype = norm.get("type")
        self.min_contribution_pct = norm.get("min_contribution_pct", 10.0)

        if stype == "self_funded":
            self.subsidy = 0.0
            self.own_fc = self.inp["own_equity_committed_lakh"]
            self.term_loan = fc - self.own_fc
            self.scheme_name = norm["name"]
            self.funding_basis = "Self-funded: FPO promoter equity + term loan (no subsidy)."

        elif stype == "capital_subsidy":
            raw = norm["subsidy_pct"] / 100.0 * self.eligible_fc
            cap = norm.get("subsidy_ceiling_lakh")
            self.subsidy = min(raw, cap) if cap else raw
            self.own_fc = self.min_contribution_pct / 100.0 * fc
            self.term_loan = fc - self.subsidy - self.own_fc
            self.scheme_name = norm["name"]
            self.funding_basis = (f"Credit-linked: {norm['name']} subsidy "
                                  f"({norm['subsidy_pct']:.0f}%) + FPO promoter "
                                  f"({self.min_contribution_pct:.0f}%) + term loan (balance).")

        elif stype == "interest_subvention":
            self.subsidy = 0.0                             # no capital grant
            self.own_fc = self.min_contribution_pct / 100.0 * fc
            self.term_loan = fc - self.own_fc
            subv = norm["interest_subvention_pct"]
            self.effective_interest_rate = max(base_rate - subv, 0.0)
            self.scheme_name = norm["name"]
            cap_note = (f"up to Rs {norm['max_loan_lakh']:.0f} L"
                        if self.term_loan <= norm["max_loan_lakh"]
                        else f"first Rs {norm['max_loan_lakh']:.0f} L of the loan")
            self.funding_basis = (f"{norm['name']}: no capital subsidy; term loan at "
                                  f"{base_rate:.1f}% less {subv:.0f}% subvention = "
                                  f"{self.effective_interest_rate:.1f}% effective ({cap_note}, "
                                  f"{norm['subvention_years']} years).")
        else:
            raise NotImplementedError(
                f"Scheme '{choice}' is a stub — add its norms/type to scheme_norms.py before use. TODO.")

    # --- working capital (traditional method) --------------------------- #
    def _working_capital(self):
        d = self.cfg["working_capital_cycle_days"]
        margin_pct = self.cfg.get("wc_margin_pct", 10) / 100.0
        self.gwc, self.wc_margin, self.wc_bank = [], [], []
        self.debtors, self.creditors, self.closing_stock = [], [], []
        for y in range(YEARS):
            rm_stock = self.rm_cost[y] / self.days * d["raw_material_days"]
            fg_stock = self.fg_close_val[y]
            stock = rm_stock + fg_stock
            creditors = self.rm_cost[y] / self.days * d["payables_days"]
            debtors = self.sales[y] / self.days * d["receivables_days"]
            paid_stock = stock - creditors
            gwc = paid_stock + debtors
            self.closing_stock.append(stock)
            self.creditors.append(creditors)
            self.debtors.append(debtors)
            self.gwc.append(gwc)
            self.wc_margin.append(gwc * margin_pct)
            self.wc_bank.append(gwc * (1 - margin_pct))
        rate = self.cfg["interest_rate_pct"] / 100.0
        # WC interest charged on the sanctioned (year-1) bank limit, flat.
        self.wc_interest = [self.wc_bank[0] * rate for _ in range(YEARS)]

    def _project_cost_and_finance(self):
        self.total_project_cost = self.fixed_capital + self.gwc[0]
        self.wc_bank_loan = self.wc_bank[0]
        self.own_equity = self.own_fc + self.wc_margin[0]     # 10% FC + WC margin
        self.total_loan = self.term_loan + self.wc_bank_loan
        min_contrib = self.min_contribution_pct / 100.0 * self.fixed_capital
        self.min_contribution_required = min_contrib
        self.min_contribution_ok = self.own_fc >= min_contrib - 0.02

    # --- P&L (branch B rent already handled) ---------------------------- #
    def _pnl(self):
        # Tax is charged on the full profit before tax (no tax-free threshold — a
        # Producer Company has no such general exemption).
        trate = self.cfg.get("tax_rate_pct", 0) / 100.0
        self.gross_profit, self.cost_of_sales = [], []
        self.pbt, self.tax, self.pat, self.cash_accrual = [], [], [], []
        for y in range(YEARS):
            cos = self.cop[y] + self.fg_open_val[y] - self.fg_close_val[y]
            self.cost_of_sales.append(cos)
            gp = self.sales[y] - cos
            self.gross_profit.append(gp)
            below = (self.staff[y] + self.tl_interest[y] + self.wc_interest[y]
                     + self.rent[y] + self.selling_adm[y])
            npbt = gp - below
            self.pbt.append(npbt)
            t = max(npbt, 0.0) * trate
            self.tax.append(t)
            self.pat.append(npbt - t)
            self.cash_accrual.append(npbt - t + self.dep[y])

    def _break_even(self):
        self.var_cost, self.fixed_cost, self.contribution, self.bep = [], [], [], []
        for y in range(YEARS):
            var = (self.rm_cost[y] + 0.85 * self.electricity[y] + 0.60 * (self.labour[y] + self.staff[y])
                   + 0.80 * self.selling_adm[y] + self.wc_interest[y] + self.repair[y] + self.packaging[y])
            fix = (0.15 * self.electricity[y] + 0.40 * (self.labour[y] + self.staff[y])
                   + self.tl_interest[y] + self.dep[y] + 0.20 * self.selling_adm[y] + self.rent[y])
            contrib = self.sales[y] - var
            self.var_cost.append(var); self.fixed_cost.append(fix); self.contribution.append(contrib)
            self.bep.append(fix / contrib * 100.0 if contrib else 0.0)

    def _balance_sheet(self):
        acc, self.cum_profit = 0.0, []
        for y in range(YEARS):
            acc += self.pat[y]; self.cum_profit.append(acc)
        self.net_worth = [self.own_equity + self.subsidy + self.cum_profit[y] for y in range(YEARS)]

    def _cash_flow(self):
        cash, self.closing_cash = 0.0, []
        for y in range(YEARS):
            cash += self.cash_accrual[y] - self.tl_principal[y]; self.closing_cash.append(cash)

    def _dscr(self):
        self.dscr = []
        for y in range(YEARS):
            cov = self.cash_accrual[y] + self.tl_interest[y]
            obl = self.tl_principal[y] + self.tl_interest[y]
            self.dscr.append(cov / obl if obl else 0.0)
        s = [d for d in self.dscr if d]
        self.avg_dscr = sum(s) / len(s) if s else 0.0

    # ------------------------------------------------------------------ #
    def _assemble_computed(self):
        L = lambda a: [_r(x) for x in a]
        self.inst["computed"] = {
            "meta": {"currency_unit": "Rs. Lakh", "years": YEARS, "revenue_model": self.revenue_model},
            "capacity_production": {
                "rm_requirement_100_kg": _r(self.rm_requirement_100, 0),
                "output_100_kg": _r(self.output_100, 0),
                "production_kg": [_r(x, 0) for x in self.production],
                "net_sale_kg": [_r(x, 0) for x in self.net_sale_kg],
                "sale_price": self.sale_price, "rm_rate": self.rm_rate,
                "sales_lakh": L(self.sales), "rm_cost_lakh": L(self.rm_cost),
                "rm_costing_method": self.rm_method,
            },
            "project_cost": {
                "machinery": _r(self.machinery), "misc_fixed_assets": _r(self.misc_assets),
                "civil": _r(self.civil), "fixed_capital": _r(self.fixed_capital),
                "working_capital_gross_y1": _r(self.gwc[0]),
                "total_project_cost": _r(self.total_project_cost),
            },
            "means_of_finance": {
                "scheme": self.scheme_name, "funding_basis": self.funding_basis,
                "own_equity": _r(self.own_equity), "subsidy": _r(self.subsidy),
                "term_loan": _r(self.term_loan), "wc_bank_loan": _r(self.wc_bank_loan),
                "total": _r(self.own_equity + self.subsidy + self.term_loan + self.wc_bank_loan),
                "min_contribution_required": _r(self.min_contribution_required),
                "min_contribution_ok": self.min_contribution_ok,
                "fallback_note": self.scheme_fallback_note,
            },
            "term_loan_schedule": {"principal": L(self.tl_principal), "interest": L(self.tl_interest),
                                   "closing_balance": L(self.tl_closing)},
            "working_capital": {"gross": L(self.gwc), "margin": L(self.wc_margin), "bank": L(self.wc_bank),
                                "debtors": L(self.debtors), "creditors": L(self.creditors),
                                "closing_stock": L(self.closing_stock)},
            "depreciation": {"annual": L(self.dep), "net_block": L(self.net_block)},
            "profitability": {
                "sales": L(self.sales), "rm_cost": L(self.rm_cost), "electricity": L(self.electricity),
                "depreciation": L(self.dep), "wages_labour": L(self.labour), "repair": L(self.repair),
                "packaging": L(self.packaging), "cost_of_sales": L(self.cost_of_sales),
                "gross_profit": L(self.gross_profit), "salary_staff": L(self.staff),
                "interest_term_loan": L(self.tl_interest), "interest_wc": L(self.wc_interest),
                "rent": L(self.rent), "selling_admin": L(self.selling_adm),
                "pbt": L(self.pbt), "tax": L(self.tax), "pat": L(self.pat),
                "cash_accrual": L(self.cash_accrual),
                # aliases kept for render/sections.py compatibility
                "wages": L(self.labour), "power": L(self.electricity), "admin": L(self.selling_adm),
                "selling": L(self.selling_adm),
            },
            "break_even": {"variable_cost": L(self.var_cost), "fixed_cost": L(self.fixed_cost),
                           "contribution": L(self.contribution), "bep_pct": L(self.bep)},
            "balance_sheet": {"net_worth": L(self.net_worth), "term_loan_closing": L(self.tl_closing),
                              "net_block": L(self.net_block), "closing_cash": L(self.closing_cash)},
            "cash_flow": {"cash_accrual": L(self.cash_accrual), "loan_repayment": L(self.tl_principal),
                          "closing_cash": L(self.closing_cash)},
            "dscr": {"annual": L(self.dscr), "average": _r(self.avg_dscr)},
        }

    def _assert_balance(self):
        mf = self.inst["computed"]["means_of_finance"]
        diff = abs(mf["total"] - self.inst["computed"]["project_cost"]["total_project_cost"])
        assert diff <= 0.05, f"means of finance {mf['total']} != project cost (diff {diff:.4f})"


def compute(instance: dict) -> dict:
    return DPRFinancialModel(instance).compute()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(config.GHEE_FIXTURE)
    with open(path) as f:
        inst = json.load(f)
    c = compute(inst)["computed"]
    G = {  # ground-truth targets from the NIFTEM ghee model DPR
        "tpc": 27.82, "subsidy": 7.66, "term_loan": 12.03, "wc_bank": 5.34,
        "rm": [51.84, 61.24, 71.28, 82.37, 94.18], "sales": [84.11, 101.00, 117.74, 135.94, 155.69],
        "dep": [3.16, 2.70, 2.30, 1.97, 1.68], "pat": [1.84, 4.26, 6.58, 9.07, 11.19],
        "dscr": [2.45, 2.15, 2.82, 3.67, 4.60], "avg_dscr": 3.14,
    }
    pc, mf, pr, cp = c["project_cost"], c["means_of_finance"], c["profitability"], c["capacity_production"]
    def line(name, got, tgt):
        print(f"  {name:22} engine={got}  target={tgt}")
    print("=== CAPITAL ===")
    line("total project cost", pc["total_project_cost"], G["tpc"])
    line("subsidy", mf["subsidy"], G["subsidy"])
    line("term loan", mf["term_loan"], G["term_loan"])
    line("wc bank", mf["wc_bank_loan"], G["wc_bank"])
    print("=== OPERATIONS ===")
    line("rm cost", cp["rm_cost_lakh"], G["rm"])
    line("sales", cp["sales_lakh"], G["sales"])
    line("depreciation", c["depreciation"]["annual"], G["dep"])
    print("=== RESULTS ===")
    line("PAT", pr["pat"], G["pat"])
    line("DSCR", c["dscr"]["annual"], G["dscr"])
    line("avg DSCR", c["dscr"]["average"], G["avg_dscr"])
