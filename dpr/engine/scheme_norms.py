"""
Government-scheme financial norms — the ONLY place the changeable subsidy /
subvention numbers live. Branch logic in financial_engine.py reads from here so
that when a scheme's rules change you edit one dict, not the engine.

Each scheme declares a `type` that selects the financing mechanism:
  * "capital_subsidy"     — upfront grant (% of fixed capital, optionally capped);
                            term loan = balance. e.g. PM-FME, MIDH.
  * "interest_subvention" — no grant; the term-loan interest rate is reduced by a
                            subvention for a number of years (up to a loan cap).
                            e.g. AIF.
  * "self_funded"         — no scheme; promoter equity + term loan only.

Every entry carries `verified_as_of` per the DPR provenance rule.
"""

SCHEME_NORMS = {
    "no_scheme": {
        "name": "Self-funded (no government scheme)",
        "type": "self_funded",
        "min_contribution_pct": 10.0,   # promoter's own share of fixed capital
        "verified_as_of": "2026-07-10",
        "source": "Client direction: loan terms as per PM-FME sample (interest/tenor/moratorium from instance config).",
    },

    "pmfme": {
        "name": "PM Formalisation of Micro Food Processing Enterprises (PM-FME)",
        "type": "capital_subsidy",
        "subsidy_pct": 35.0,             # % of eligible (fixed) capital
        "subsidy_ceiling_lakh": 10.0,    # hard cap on the subsidy
        "min_contribution_pct": 10.0,    # promoter contribution as % of fixed capital
        "eligible_cost_basis": "fixed_capital",  # machinery + misc fixed assets (+ civil)
        "verified_as_of": "2026-07-08",
        "source": "NIFTEM PM-FME model DPRs (35% subsidy / Rs 10 L cap / 10% promoter).",
    },

    "midh": {
        "name": "Mission for Integrated Development of Horticulture (MIDH)",
        "type": "capital_subsidy",
        "subsidy_pct": 40.0,             # 40-50% state-specific; 40% used (confirm per State Horticulture Mission)
        "subsidy_ceiling_lakh": None,    # no fixed cap given — component/state-specific — CONFIRM
        "min_contribution_pct": 10.0,    # ASSUMED (not specified) — confirm
        "eligible_cost_basis": "fixed_capital",
        "applicable_to": "horticulture (fruits, vegetables, flowers, spices, mushrooms, medicinal plants)",
        "verified_as_of": "2026-07-10",
        "source": "NAFPO Academy MIDH summary (40-50% subsidy via State Horticulture Missions).",
        "assumptions": "subsidy 40% (range 40-50 state-specific); no ceiling and 10% promoter assumed — confirm with State Horticulture Mission.",
    },

    "aif": {
        "name": "Agriculture Infrastructure Fund (AIF)",
        "type": "interest_subvention",
        "interest_subvention_pct": 3.0,  # rebate on term-loan interest
        "max_loan_lakh": 200.0,          # Rs 2 crore — subvention applies up to this loan size
        "subvention_years": 7,           # duration of the subvention
        "min_contribution_pct": 10.0,    # ASSUMED bank margin (AIF is a loan, not a grant) — confirm
        "min_lease_years": 7,            # eligibility: owned land or 7-year lease
        "verified_as_of": "2026-07-10",
        "source": "NAFPO Academy AIF summary (loan up to Rs 2 cr, 3% interest subvention, 7 years).",
        "assumptions": "10% promoter margin assumed; no capital subsidy (interest subvention only).",
    },

    # --- Still a stub: rules intentionally NOT fabricated ---
    "smam": {
        "name": "Sub-Mission on Agricultural Mechanization (SMAM)",
        "type": None,
        "implemented": False,
        "verified_as_of": None,
    },
}
