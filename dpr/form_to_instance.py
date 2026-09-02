"""
Convert a website form payload (schema/form_payload_example.json structure) +
its business-line template (data/templates/<business_line>.json) into an engine
instance JSON that run_dpr.py consumes.

Mapping:
  form payload  -> FPO-specific facts (identity, land, capacity, own funds,
                   scheme, capex, prices)
  template      -> technical norms (wastage, misc-assets default, machinery
                   item list, narrative process steps, full config block)

Usage:
    ./venv/bin/python form_to_instance.py data/payloads/ghee_payload.json
    # -> data/generated/<commodity>_instance.json
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

from dpr import config
TEMPLATES = config.TEMPLATES_DIR
OUT_DIR = config.GENERATED_DIR

LAKH = 1e5


CROP_ROUTED_LINES = {
    "rice_dal_flour_mill": [
        (("paddy", "rice"), "rice_mill"),
        (("wheat", "atta", "maida"), "flour_mill"),
        (("gram", "chana", "chickpea", "pulse", "dal", "lentil", "tur", "arhar",
          "moong", "mung", "urad", "masoor", "pigeon pea", "bengal"), "dal_mill"),
    ],
    "processing_unit": [
        (("milk", "dairy", "ghee", "butter", "curd"), "processing_unit_ghee"),
        (("turmeric", "chilli", "chili", "coriander", "cumin", "ginger", "spice",
          "pepper", "cardamom", "fenugreek", "fennel", "clove"), "processing_unit_spices"),
        (("amla", "aonla", "mango", "guava", "pineapple", "orange", "citrus",
          "pomegranate", "grape", "litchi", "fruit", "juice"), "processing_unit_fruit_juice"),
    ],
}


def resolve_business_line(s0: dict) -> tuple[str, str | None]:
    """Map the form's business_line to a template stem, and return the commodity
    that selected it. For combined options (Mill, Processing Unit) the template is
    chosen from the explicit `commodity` field (e.g. 'Turmeric' -> spices, 'Pulses /
    Dal' -> dal_mill); top_crops is only a fallback for older payloads without one.
    Direct template names pass through unchanged (matched crop is None -> caller
    falls back to the commodity field / top_crops[0])."""
    bl = s0["business_line"]
    routes = CROP_ROUTED_LINES.get(bl)
    if routes is None:
        return bl, None
    candidates = [s0.get("commodity")] + list(s0.get("top_crops", []))
    for crop in candidates:
        if not crop:
            continue
        cl = str(crop).lower()
        for keywords, tpl_name in routes:
            if any(k in cl for k in keywords):
                return tpl_name, crop
    raise ValueError(
        f"business_line '{bl}' but neither commodity {s0.get('commodity')!r} nor "
        f"top_crops {s0.get('top_crops')!r} matched a commodity template.")


def convert_nested(payload: dict) -> dict:
    s0, s1 = payload["stage0_basic_info"], payload["stage1_land_business"]
    s2, costs = payload["stage2_scheme"], payload["project_costs_and_prices"]

    business_line, matched_crop = resolve_business_line(s0)
    tpl_path = TEMPLATES / f"{business_line}.json"
    if not tpl_path.exists():
        raise FileNotFoundError(f"No business-line template: {tpl_path}")
    tpl = json.loads(tpl_path.read_text())
    tdef = tpl["technical_defaults"]
    today = date.today().isoformat()

    # Product identity: "category" templates (one template, many actual products -
    # e.g. fruit juice, oil, spices) carry a `product_label` like "{crop} Juice".
    # The specific commodity comes from the explicit `commodity` field (as on the
    # website form); top_crops is only a fallback for older payloads without one.
    # The product description is handed to the LLM (product_intro / raw_material_note
    # -> None) so it is written for the real product, not the template's hardcoded one.
    top_crops = s0.get("top_crops") or []
    primary_crop = (s0.get("commodity") or s1.get("commodity")
                    or matched_crop or (top_crops[0] if top_crops else None))
    label_fmt = tpl.get("product_label")
    if label_fmt and primary_crop:
        product_name = label_fmt.format(crop=str(primary_crop).strip().title())
        report_title = "DETAILED PROJECT REPORT - " + product_name.upper()
        llm_writes_product = True
    else:
        product_name = tpl["commodity"]
        report_title = tpl["report_title"]
        llm_writes_product = False

    # revenue_model: "processing" (RM -> product) or "service" (CHC / warehouse:
    # capacity is rented, no raw-material -> product transformation).
    revmodel = tpl.get("revenue_model", "processing")

    # --- capacity ---
    #   processing: form gives input/day + operating days; engine wants per-hour.
    #   service:    form gives annual service units (machine-hours, MT-storage).
    cap = s1["target_capacity"]
    if cap["unit"] != tpl["capacity_input_unit"]:
        raise ValueError(f"capacity unit {cap['unit']!r} != template's {tpl['capacity_input_unit']!r}")
    cap_mode = tpl.get("capacity_mode", "per_day")
    if revmodel == "service":
        capacity_block = {
            "annual_service_units": cap["annual_units"],
            "working_days_per_annum": cap.get("operating_days_per_year", 300),
        }
    elif cap_mode == "annual_throughput":
        # Batch processing (nursery/hardening): capacity is the annual saleable
        # output. No wastage applied (same output model as convert_website).
        capacity_block = {
            "annual_input_units": cap["annual_units"],
            "working_days_per_annum": cap.get("operating_days_per_year", 300),
            "wastage_pct": 0,
            "pack_size_kg": tdef.get("pack_size_kg", 1),
        }
    else:
        per_day_units = cap["input_per_day"] * tpl["kg_per_capacity_unit"]
        per_hr = per_day_units / tpl["shift_hours"]
        capacity_block = {
            "per_hr_kg": per_hr,
            "shift_hours": tpl["shift_hours"],
            "working_days_per_annum": cap["operating_days_per_year"],
            "wastage_pct": 0,                       # output model — same as convert_website
            "pack_size_kg": tdef["pack_size_kg"],
        }

    # --- config: use the SAME electricity mechanism as convert_website (monthly x12).
    #     Derive the monthly figure from the template's connected load so the amount
    #     is unchanged, but the engine takes the same code path as production. ---
    cfg_nested = json.loads(json.dumps(tpl["config"]))
    _pw = cfg_nested.get("power", {})
    _days = capacity_block.get("working_days_per_annum", 300)
    cfg_nested["electricity_monthly_inr"] = round(
        _pw.get("units_per_day", 0) * _pw.get("rate_per_unit", 0) * _days / 12, 2)

    # --- scheme: payload choice wins; else the business line's default scheme
    #     (e.g. vermicompost -> midh); no_scheme if the FPO opted out. ---
    if s2.get("apply_for_scheme"):
        scheme = s2.get("scheme_choice") or tpl.get("default_scheme") or "no_scheme"
    else:
        scheme = "no_scheme"

    # --- location & narrative ---
    loc = s0.get("location", {})
    location = ", ".join(x for x in [loc.get("village"), loc.get("block"),
                                     loc.get("district"), loc.get("state")] if x)
    buyers = s1.get("anticipated_buyers") or []

    machinery_items = costs.get("machinery_items_optional") or tpl["machinery_items_indicative"]
    misc_lakh = costs.get("other_fixed_assets_lakh")
    if misc_lakh is None:
        misc_lakh = tdef["misc_fixed_assets_lakh"]

    prov = lambda src: {"source": src, "verified_as_of": today}

    instance = {
        "meta": {
            "commodity": product_name,
            "report_title": report_title,
            "revenue_model": revmodel,
            "product_unit": tpl.get("product_unit", "kg"),
            "synthetic": True,
            "prepared_by": "FPO DPR Generator (from website form payload)",
            "created_as_of": today,
        },
        "narrative": {
            "fpo_name": s0.get("fpo_name"),
            "location": location or None,
            "members": s0.get("total_members"),
            "product_intro": None if llm_writes_product else tpl["narrative"]["product_intro"],
            "product_uses_benefits": None,
            "market_potential": None,
            "raw_material_note": None if llm_writes_product else tpl["narrative"]["raw_material_note"],
            "process_steps": tpl["narrative"]["process_steps"],
            "value_chain": None,
            "market_text": ("Anticipated buyers: " + ", ".join(buyers)) if buyers else None,
            "risk_text": None,
            "governance": (f"Registered {s0['year_of_registration']}; annual turnover "
                           f"Rs. {s0['annual_turnover_inr']/LAKH:.0f} lakh."
                           if s0.get("year_of_registration") else None),
            "employment": (f"Expected direct jobs: {s1['expected_direct_jobs']}."
                           if s1.get("expected_direct_jobs") else None),
            "sustainability": None,
        },
        "inputs": {
            # normalise any free-form land label to the engine's {owned, leased}
            "land_status": ("leased" if any(k in str(s1["land_status"]).lower()
                                            for k in ("leas", "rent")) else "owned"),
            "rent_per_month_inr": costs.get("rent_per_month_inr") or 0,
            "capex": {
                "machinery_inr": costs["machinery_total_lakh"] * LAKH,
                "machinery_items": machinery_items,
                "civil_construction_lakh": costs.get("civil_construction_lakh") or 0,
                "misc_fixed_assets_lakh": misc_lakh,
            },
            "prices": {
                "rm_costing_method": costs["raw_material"]["costing_method"],
                "raw_material_price": (costs["raw_material"]["value"] / 100.0
                                       if costs["raw_material"]["costing_method"] == "pct_of_revenue"
                                       else costs["raw_material"]["value"]),
                "sale_price": costs["product_selling_price_inr_per_unit"],
            },
            "capacity": capacity_block,
            "own_equity_committed_lakh": s1["own_funds_committed_inr"] / LAKH,
            "scheme": {"choice": scheme},
            "provenance": {
                "capex.machinery_inr": prov("form: machinery_total_lakh (vendor quote)"),
                "capex.misc_fixed_assets_lakh": prov(
                    "form: other_fixed_assets_lakh" if costs.get("other_fixed_assets_lakh") is not None
                    else f"template default ({tpl['business_line']})"),
                "rent_per_month_inr": prov("form: rent_per_month_inr"),
                "prices.raw_material_price": prov("form: raw_material"),
                "prices.sale_price": prov("form: product_selling_price_inr_per_unit"),
                **({"capacity.annual_service_units": prov("form: target_capacity (annual service units)")}
                   if revmodel == "service" else
                   {"capacity.per_hr_kg": prov("form: target_capacity (input/day / shift hours)"),
                    "capacity.wastage_pct": prov(f"template default ({tpl['business_line']})")}),
                "own_equity_committed_lakh": prov("form: own_funds_committed_inr"),
            },
        },
        "config": cfg_nested,
        "computed": {},
    }
    return instance


# ===================================================================== #
#  WEBSITE (flat) payload adapter — the PRODUCTION path
#  Consumes the NAFPO "FPO Document Builder" flat `fields` payload
#  (docs/schema/FPO-Document-Builder-schema.json) and produces an engine
#  instance. Defensive by design: never raises on bad/blank input — every
#  field falls back to a calibrated per-line default so a viable DPR is
#  always produced. The engine still owns all the arithmetic.
# ===================================================================== #

# Form businessLine -> template stem (the 12 client-confirmed lines).
BL_TO_TEMPLATE = {
    "Rice Mill": "rice_mill",
    "Dhal Mill": "dal_mill",
    "Flour Mill": "flour_mill",
    "Oil Extraction": "oil_extraction_unit",
    "Ghee Processing": "processing_unit_ghee",
    "Spice Processing": "processing_unit_spices",
    "Juice Processing": "processing_unit_fruit_juice",
    "Fruit & Vegetable Grading": "veg_fruit_grading_sorting_unit",
    "Nursery Hardening": "nursery_tissue_culture_hardening_unit",
    "Vermicompost": "vermicompost_organic_fertilizer_unit",
    "Custom Hiring Centre": "custom_hiring_centre",
    "Storage / Warehouse": "warehouse_cold_storage",
}

# Lines whose product identity comes from the `commodity` field (LLM writes the
# product intro for that commodity). value = f-string on the cleaned commodity.
COMMODITY_VARIABLE = {
    "oil_extraction_unit": "{c} Oil",
    "processing_unit_fruit_juice": "{c} Juice",
    "processing_unit_spices": "{c}",
}

# Per-line engine defaults used when a form field is blank. Capacity is modelled
# as OUTPUT THROUGHPUT in the line's own unit at the line's own price (this is how
# the form's reference_data frames it: units/day x price/unit). `basis`:
#   "throughput_per_day" -> annual = capacity x working_days (wastage 0 in engine)
#   "annual"             -> capacity is used as the annual figure directly
#                           (service lines; client decision: the "units/day" label
#                            on the form is cosmetic).
# Sources: schema reference_data for the 8 lines that carry defaults; our
# calibrated base payloads for Dhal/Flour/Ghee/Spice (reference_data = null there).
LINE_DEFAULTS = {
    "rice_mill":            dict(unit="kg",           cap=375,   days=300, price=47,    rm=("unit", 22),   basis="throughput_per_day"),
    "dal_mill":             dict(unit="kg",           cap=1600,  days=300, price=88,    rm=("unit", 67.5), basis="throughput_per_day"),
    "flour_mill":           dict(unit="kg",           cap=6000,  days=300, price=28,    rm=("unit", 22),   basis="throughput_per_day"),
    "oil_extraction_unit":  dict(unit="Ltr",          cap=237.5, days=300, price=140,   rm=("unit", 35),   basis="throughput_per_day"),
    "processing_unit_ghee": dict(unit="kg",           cap=240,   days=300, price=330,   rm=("unit", 180),  basis="throughput_per_day"),
    "processing_unit_spices": dict(unit="kg",         cap=300,   days=300, price=165,   rm=("unit", 95),   basis="throughput_per_day"),
    "processing_unit_fruit_juice": dict(unit="KL",    cap=0.225, days=310, price=98000, rm=("pct", 58),    basis="throughput_per_day"),
    "veg_fruit_grading_sorting_unit": dict(unit="kg", cap=575,   days=300, price=16.2,  rm=("unit", 9.8),  basis="throughput_per_day"),
    "nursery_tissue_culture_hardening_unit": dict(unit="plant", cap=145, days=310, price=16.5, rm=("unit", 6), basis="throughput_per_day"),
    "vermicompost_organic_fertilizer_unit": dict(unit="MT",  cap=0.29,  days=310, price=4880,  rm=("unit", 132),  basis="throughput_per_day"),
    "custom_hiring_centre": dict(unit="machine-hour", cap=5000,  days=305, price=680,   rm=("pct", 30),    basis="annual"),
    "warehouse_cold_storage": dict(unit="MT-year",    cap=2000,  days=365, price=2450,  rm=("pct", 5),     basis="annual"),
}

# schemeName -> engine scheme choice. "Others" -> an unmodelled key so the engine
# falls back to self-funded AND surfaces a note (see financial_engine fallback).
SCHEME_MAP = {"PMFME": "pmfme", "AIF": "aif", "MIDH": "midh", "Others": "others"}

# Manpower role -> which cost bucket. Worker/Labour/Operator etc. are COGS labour;
# everyone else (supervisor, sales, security, admin, unknown) is admin staff.
_LABOUR_ROLE_KEYS = ("worker", "labour", "labor", "operator", "helper", "packer",
                     "loader", "machine", "plant", "process", "production")


def _num(x, default=0.0):
    """Parse a number from anything the form might send (int/float/str with commas,
    'Rs', '%', blanks). Non-numeric / None -> default. Never raises."""
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return float(x) if x == x else default          # x==x rejects NaN
    if x is None:
        return default
    s = str(x).strip()
    if not s:
        return default
    m = re.search(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    try:
        return float(m.group(0)) if m else default
    except (ValueError, AttributeError):
        return default


def _clean_commodity(commodity, other) -> str:
    """Turn the dropdown value into a product word: 'Other' -> free text; otherwise
    the first token before a slash ('Groundnut / Oilseeds' -> 'Groundnut',
    'Paddy / Rice' -> 'Paddy')."""
    raw = (other if str(commodity).strip().lower() == "other" else commodity) or ""
    raw = str(raw).split("/")[0].strip()
    return raw.title() if raw else "Product"


def _classify_manpower(rows: list, tpl: dict):
    """Split the form's manpower rows into labour (COGS) and staff (admin) wage
    tables. Empty / unusable -> the template's own tables."""
    labour, staff = [], []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        role = str(r.get("role", "")).strip()
        count = int(_num(r.get("count"), 0))
        wage = _num(r.get("monthly_wage_inr"), 0)
        if count <= 0 or wage <= 0:
            continue
        entry = {"role": role or "Staff", "count": count, "monthly_wage_inr": wage}
        bucket = labour if any(k in role.lower() for k in _LABOUR_ROLE_KEYS) else staff
        bucket.append(entry)
    lab = labour or [dict(w) for w in tpl["config"]["labour_wage_table"]]
    stf = staff or [dict(w) for w in tpl["config"]["staff_wage_table"]]
    return lab, stf


def convert_website(payload: dict) -> dict:
    """Convert the flat NAFPO form payload to an engine instance. Accepts either the
    raw `fields` dict or a `{document_type, fields}` wrapper."""
    f = payload.get("fields", payload) if isinstance(payload, dict) else {}
    today = date.today().isoformat()

    # --- business line -> template (unknown line -> warehouse as a safe default) --
    bl = str(f.get("businessLine", "")).strip()
    stem = BL_TO_TEMPLATE.get(bl, "warehouse_cold_storage")
    tpl = json.loads((TEMPLATES / f"{stem}.json").read_text())
    dflt = LINE_DEFAULTS[stem]
    cfg = json.loads(json.dumps(tpl["config"]))          # deep copy; overlay form values

    # --- product identity (commodity-variable lines let the LLM write the intro) --
    commodity_word = _clean_commodity(f.get("commodity"), f.get("commodityOther"))
    fmt = COMMODITY_VARIABLE.get(stem)
    if fmt:
        product_name = fmt.format(c=commodity_word).strip()
        report_title = "DETAILED PROJECT REPORT - " + product_name.upper()
        llm_writes_product = True
    else:
        product_name = tpl.get("commodity")
        report_title = tpl.get("report_title", "DETAILED PROJECT REPORT")
        llm_writes_product = False
    revmodel = tpl.get("revenue_model", "processing")

    # --- capacity (output-throughput model; see LINE_DEFAULTS) --------------------
    cap_val = _num(f.get("capacity"), dflt["cap"])
    if cap_val <= 0:
        cap_val = dflt["cap"]
    days = int(_num(f.get("workingDays"), dflt["days"]) or dflt["days"])
    days = max(1, min(days, 366))
    # Capacity is the finished-product throughput per day, priced per output unit.
    # No wastage/recovery is applied (client decision): the entered capacity IS the
    # saleable output, so input == output across all lines.
    pack = tpl.get("technical_defaults", {}).get("pack_size_kg", 1)
    if revmodel == "service" or dflt["basis"] == "annual":
        capacity_block = {"annual_service_units": cap_val, "working_days_per_annum": days}
    elif tpl.get("capacity_mode") == "annual_throughput":
        capacity_block = {"annual_input_units": cap_val * days, "working_days_per_annum": days,
                          "wastage_pct": 0, "pack_size_kg": pack}
    else:
        shift = tpl.get("shift_hours") or 8
        capacity_block = {"per_hr_kg": cap_val / shift, "shift_hours": shift,
                          "working_days_per_annum": days, "wastage_pct": 0, "pack_size_kg": pack}

    # --- prices & raw-material costing --------------------------------------------
    sale_price = _num(f.get("salePrice"), dflt["price"])
    if sale_price <= 0:
        sale_price = dflt["price"]
    rm = f.get("rmcost") if isinstance(f.get("rmcost"), dict) else {}
    rm_mode = str(rm.get("mode", "")).strip().lower()
    rm_value = _num(rm.get("value"), None)
    if rm_mode not in ("unit", "pct") or rm_value is None:
        rm_mode, rm_value = dflt["rm"]                   # template/line default
    if rm_mode == "pct":
        rm_method, rm_price = "pct_of_revenue", max(0.0, rm_value) / 100.0
    else:
        rm_method, rm_price = "per_unit", max(0.0, rm_value)

    # --- capex: machinery table (sum rows), civil (always), misc ------------------
    mach_rows = f.get("machinery") if isinstance(f.get("machinery"), list) else []
    items = [{"name": str(r.get("item", "Equipment")), "inr": _num(r.get("cost_inr"), 0)}
             for r in mach_rows if isinstance(r, dict) and _num(r.get("cost_inr"), 0) > 0]
    if items:
        machinery_inr = sum(i["inr"] for i in items)
    else:                                                # blank -> template machinery
        items = tpl["machinery_items_indicative"]
        machinery_inr = sum(i["inr"] for i in items if isinstance(i, dict))
    civil_lakh = max(0.0, _num(f.get("civilCost"), 0) / LAKH)
    misc_lakh = _num(f.get("otherFixedAssets"), None)
    misc_lakh = tpl["technical_defaults"]["misc_fixed_assets_lakh"] if misc_lakh in (None, 0) \
        else max(0.0, misc_lakh / LAKH)

    # --- land / rent --------------------------------------------------------------
    land = "leased" if "leas" in str(f.get("landStatus", "")).lower() \
        or "rent" in str(f.get("landStatus", "")).lower() else "owned"
    rent = max(0.0, _num(f.get("monthlyRent"), 0)) if land == "leased" else 0.0

    # --- scheme -------------------------------------------------------------------
    choice_raw = str(f.get("schemeChoice", "")).strip().lower()
    if choice_raw == "yes":
        nm = str(f.get("schemeName", "")).strip()
        scheme = SCHEME_MAP.get(nm)
        if not scheme:               # any unlisted / free-text scheme -> unknown key so the
            scheme = re.sub(r"[^0-9A-Za-z]+", "_", nm).strip("_") or "others"   # engine falls back + notes it
    else:                                                # "No" / "Not sure" / blank
        scheme = "no_scheme"

    # --- config overrides from the form -------------------------------------------
    cfg["annual_price_escalation_pct"] = _num(f.get("priceEscalation"), cfg.get("annual_price_escalation_pct", 5)) or 5
    cfg["electricity_monthly_inr"] = max(0.0, _num(f.get("electricityCost"), 15000))
    loan = f.get("loan") if isinstance(f.get("loan"), dict) else {}
    interest = _num(loan.get("interest_pct"), cfg.get("interest_rate_pct", 11))
    cfg["interest_rate_pct"] = interest if interest > 0 else cfg.get("interest_rate_pct", 11)
    tenure = int(_num(loan.get("tenure_months"), 0))
    if tenure <= 12:                                     # form default 12 == no repayment window -> sane default
        tenure = 60
    cfg["loan_tenor_months"] = tenure
    morat = int(_num(loan.get("moratorium_months"), cfg.get("moratorium_months", 12)))
    cfg["moratorium_months"] = max(0, min(morat, tenure - 1))
    cfg["labour_wage_table"], cfg["staff_wage_table"] = _classify_manpower(f.get("manpower"), tpl)

    # --- narrative (applicant/contact block + LLM-authored fields) ----------------
    year = f.get("year")
    turnover = str(f.get("turnover") or "").strip()
    gov = None
    if year:
        gov = f"Registered {int(_num(year, 0))}" + (f"; annual turnover {turnover}." if turnover else ".")
    buyers = str(f.get("buyers") or "").strip()
    jobs = str(f.get("jobs") or "").strip()

    prov = lambda src: {"source": src, "verified_as_of": today}
    instance = {
        "meta": {
            "commodity": product_name,
            "report_title": report_title,
            "revenue_model": revmodel,
            "product_unit": dflt["unit"],
            "synthetic": False,
            "prepared_by": "FPO DPR Generator (NAFPO FPO Document Builder form)",
            "created_as_of": today,
        },
        "narrative": {
            "fpo_name": (str(f.get("fpoName")).strip() or None) if f.get("fpoName") else None,
            "location": (str(f.get("location")).strip() or None) if f.get("location") else None,
            "registration_no": (str(f.get("registrationNo")).strip() or None) if f.get("registrationNo") else None,
            "ceo_name": (str(f.get("ceoName")).strip() or None) if f.get("ceoName") else None,
            "mobile": (str(f.get("mobile")).strip() or None) if f.get("mobile") else None,
            "email": (str(f.get("email")).strip() or None) if f.get("email") else None,
            "site_address": (str(f.get("siteAddress")).strip() or None) if f.get("siteAddress") else None,
            "members": int(_num(f.get("members"), 0)) or None,
            "product_intro": None if llm_writes_product else tpl["narrative"]["product_intro"],
            "product_uses_benefits": None,
            "market_potential": None,
            "raw_material_note": None if llm_writes_product else tpl["narrative"]["raw_material_note"],
            "process_steps": tpl["narrative"]["process_steps"],
            "value_chain": None,
            "market_text": (f"Anticipated buyers: {buyers}" if buyers else None),
            "risk_text": None,
            "governance": gov,
            "employment": (f"Expected direct jobs: {jobs}." if jobs else None),
            "sustainability": None,
            "top_crops_context": (str(f.get("crops")).strip() or None) if f.get("crops") else None,
            # the FPO's selected commodity — used as the product name on Oil/Juice/Spice,
            # and as a crop/focus context on every other line (shown in §1 + fed to the LLM)
            "commodity_focus": commodity_word or None,
        },
        "inputs": {
            "land_status": land,
            "rent_per_month_inr": rent,
            "capex": {
                "machinery_inr": machinery_inr,
                "machinery_items": items,
                "civil_construction_lakh": civil_lakh,
                "misc_fixed_assets_lakh": misc_lakh,
            },
            "prices": {
                "rm_costing_method": rm_method,
                "raw_material_price": rm_price,
                "sale_price": sale_price,
            },
            "capacity": capacity_block,
            "own_equity_committed_lakh": max(0.0, _num(f.get("ownFunds"), 0) / LAKH),
            "scheme": {"choice": scheme},
            "provenance": {
                "capex.machinery_inr": prov("form: machinery table (sum of rows)"
                                            if items and items is not tpl["machinery_items_indicative"]
                                            else f"template default ({stem})"),
                "prices.sale_price": prov("form: salePrice" if _num(f.get("salePrice"), 0) > 0
                                          else f"line default ({stem})"),
                "prices.raw_material_price": prov("form: rmcost" if rm.get("value") not in (None, "")
                                                  else f"line default ({stem})"),
                "capacity": prov("form: capacity" if _num(f.get("capacity"), 0) > 0
                                 else f"line default ({stem})"),
                "own_equity_committed_lakh": prov("form: ownFunds"),
            },
        },
        "config": cfg,
        "computed": {},
    }
    return instance


def convert(payload: dict) -> dict:
    """Dispatch on payload shape: the internal nested payload (dataset generation)
    vs. the flat NAFPO website payload (production)."""
    if isinstance(payload, dict) and "stage0_basic_info" in payload:
        return convert_nested(payload)
    return convert_website(payload)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: form_to_instance.py <payload.json> [out.json]", file=sys.stderr)
        return 1
    payload = json.loads(Path(sys.argv[1]).read_text())
    instance = convert(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in instance["meta"]["commodity"]).strip("_")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR / f"{slug}_instance.json"
    out.write_text(json.dumps(instance, indent=2, ensure_ascii=False))
    print(f"[ok] {sys.argv[1]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
