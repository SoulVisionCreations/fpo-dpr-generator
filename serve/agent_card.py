"""Discovery document: what this agent is, what it accepts, what it returns.

`GET /info` is the integrator-facing card; `GET /v1/info` is its operational twin
(vLLM version, engine state, in-flight count). Two endpoints because the readers
differ — a partner wiring up the API should not have to read engine internals to
find the field list, and an ops dashboard should not have to parse a field list
to read the engine state.

Everything derivable is derived: the business-line table from
`form_to_instance.LINE_DEFAULTS`, the scheme list from `SCHEME_MAP`, the output
shape from the Pydantic response model. Hand-copying any of them is how the
capacity-basis error got into a client-facing example and produced a DPR with a
negative DSCR — documented once, never re-checked against the code.
"""
from __future__ import annotations

from typing import Any

from dpr import config
from dpr.form_to_instance import BL_TO_TEMPLATE, LINE_DEFAULTS, SCHEME_MAP

NAME = "fpo-dpr-generator"

DESCRIPTION = (
    "Generates a NABARD / NIFTEM (PM-FME) format Detailed Project Report for an "
    "Indian Farmer Producer Organisation from a filled enterprise form, across "
    "twelve agri-enterprise business lines. "
    "A deterministic Python engine computes every financial figure — project "
    "cost, means of finance, profitability, DSCR, repayment schedule — and a "
    "fine-tuned language model writes only the narrative sections around them. "
    "The model never emits a number: it is given table placeholders, and every "
    "figure in the returned document is re-checked against the engine before the "
    "response is sent. See `verification.ok`."
)

# The engine's own vocabulary ("throughput_per_day") is not what a form-filler
# thinks in. Translate once, here.
_BASIS_HUMAN = {"throughput_per_day": "per day", "annual": "per year"}


def business_lines() -> list[dict[str, Any]]:
    """The twelve accepted `businessLine` values, each with its capacity unit.

    The per-day vs per-year split is the single most common integration error:
    a Storage / Warehouse capacity sent as a daily figure produces a DPR that
    computes and balances but reports a negative DSCR — plausible, and wrong.
    Publishing the basis per line is the cheapest way to stop that happening.
    """
    out = []
    for label, stem in BL_TO_TEMPLATE.items():
        d = LINE_DEFAULTS[stem]
        out.append({
            "businessLine": label,
            "capacity_unit": d["unit"],
            "capacity_basis": _BASIS_HUMAN[d["basis"]],
            "capacity_default": d["cap"],
            "salePrice_unit": f"INR per {d['unit']}",
            "salePrice_default": d["price"],
            "workingDays_default": d["days"],
        })
    return out


# Only `businessLine` is required. Every other field falls back to a line-standard
# default, and each substitution is reported in `defaults_applied` — so the list
# below marks what is *read*, not what must be sent.
INPUT_FIELDS: tuple[dict[str, Any], ...] = (
    # --- identity -------------------------------------------------------------
    {"name": "fpoName", "type": "string", "group": "identity"},
    {"name": "location", "type": "string", "group": "identity",
     "notes": "District, State."},
    {"name": "siteAddress", "type": "string", "group": "identity"},
    {"name": "registrationNo", "type": "string", "group": "identity"},
    {"name": "year", "type": "integer", "group": "identity",
     "notes": "Year of registration."},
    {"name": "members", "type": "integer", "group": "identity"},
    {"name": "turnover", "type": "string", "group": "identity",
     "notes": "Latest annual turnover, as text; narrative only, not computed on."},
    {"name": "crops", "type": "string", "group": "identity",
     "notes": "Comma-separated."},

    # --- enterprise -----------------------------------------------------------
    {"name": "businessLine", "type": "string", "group": "enterprise",
     "required": True,
     "notes": "One of the twelve labels in `business_lines`, exactly. Anything "
              "else returns 422 listing the valid values."},
    {"name": "commodity", "type": "string", "group": "enterprise",
     "notes": "On commodity-variable lines (oil, juice, spices) this names the "
              "product and the model writes the product introduction for it."},
    {"name": "commodityOther", "type": "string", "group": "enterprise",
     "notes": "Free text when `commodity` is 'Other'."},

    # --- capacity and revenue -------------------------------------------------
    {"name": "capacity", "type": "number", "group": "capacity",
     "notes": "UNIT AND BASIS DEPEND ON `businessLine` — see `business_lines`. "
              "Most lines are per day; Storage / Warehouse and Custom Hiring "
              "Centre are per year."},
    {"name": "workingDays", "type": "integer", "group": "capacity",
     "notes": "Operating days per year."},
    {"name": "salePrice", "type": "number", "group": "capacity",
     "notes": "INR per capacity unit, same unit as `capacity`."},
    {"name": "priceEscalation", "type": "number", "group": "capacity",
     "unit": "%/year", "notes": "Default 5."},
    {"name": "pack_size_kg", "type": "number", "group": "capacity"},

    # --- capital cost ---------------------------------------------------------
    {"name": "landStatus", "type": "string", "group": "capex",
     "notes": "'Owned' or 'Leased'."},
    {"name": "monthlyRent", "type": "number", "group": "capex", "unit": "INR/month",
     "notes": "Counted only when `landStatus` is leased."},
    {"name": "civilCost", "type": "number", "group": "capex", "unit": "INR"},
    {"name": "otherFixedAssets", "type": "number", "group": "capex", "unit": "INR"},
    {"name": "machinery", "type": "array", "group": "capex",
     "item": {"item": "string", "cost_inr": "number"},
     "notes": "Rows with cost_inr <= 0 are dropped. An empty list falls back to "
              "the line's indicative machinery table."},

    # --- operating cost -------------------------------------------------------
    {"name": "rmcost", "type": "object", "group": "opex",
     "item": {"mode": "'unit' | 'pct'", "value": "number"},
     "notes": "Raw material cost: 'unit' = INR per capacity unit, 'pct' = "
              "percent of revenue. Falls back to the line default."},
    {"name": "electricityCost", "type": "number", "group": "opex",
     "unit": "INR/month", "notes": "Default 15000."},
    {"name": "manpower", "type": "array", "group": "opex",
     "item": {"role": "string", "count": "integer", "monthly_wage_inr": "number"},
     "notes": "Roles containing worker/labour/operator/helper/packer are costed "
              "as COGS labour; everything else as admin staff."},
    {"name": "jobs", "type": "string", "group": "opex",
     "notes": "Employment generated; narrative only."},

    # --- market and finance ---------------------------------------------------
    {"name": "buyers", "type": "string", "group": "finance",
     "notes": "Narrative only."},
    {"name": "ownFunds", "type": "number", "group": "finance", "unit": "INR"},
    {"name": "schemeChoice", "type": "string", "group": "finance",
     "notes": "'Yes' or 'No'."},
    {"name": "schemeName", "type": "string", "group": "finance",
     "notes": "See `schemes`. An unmodelled scheme is self-funded in the "
              "computation and says so in `warnings`."},
    {"name": "loan", "type": "object", "group": "finance",
     "item": {"interest_pct": "number", "tenure_months": "integer",
              "moratorium_months": "integer"},
     "notes": "Tenure of 12 months or less is treated as unset."},

    # --- accepted, but do not send -------------------------------------------
    {"name": "ceoName", "type": "string", "group": "contact", "discouraged": True},
    {"name": "mobile", "type": "string", "group": "contact", "discouraged": True},
    {"name": "email", "type": "string", "group": "contact", "discouraged": True},
)

CONTACT_NOTE = (
    "ceoName, mobile and email are accepted but reach the model's narrative "
    "context. The DPR does not need them — omit them."
)


def card() -> dict[str, Any]:
    """The full discovery document."""
    # Imported here, not at module scope: serve.schemas pulls in Pydantic, and
    # keeping this module importable on its own makes it testable without the app.
    from serve.schemas import GenerateResponse

    return {
        "name": NAME,
        "description": DESCRIPTION,
        "version": "0.1.0",
        "endpoints": {
            "generate": {
                "method": "POST",
                "paths": ["/run", "/dpr", "/v1/dpr:generate"],
                "notes": "Three names for one handler. /run is the generic verb, "
                         "/dpr is what the UAI connector is pointed at, "
                         "/v1/dpr:generate is the native name.",
            },
            "info": {"method": "GET", "paths": ["/info"], "notes": "This document."},
            "health": {"method": "GET", "paths": ["/health"], "notes": "Liveness."},
            "ready": {"method": "GET", "paths": ["/ready"],
                      "notes": "Readiness. 200 only once the weights are loaded; "
                               "a cold start takes 3-5 minutes."},
        },
        "input": {
            "format": "JSON. Either the bare form, or wrapped as "
                      "{\"payload\": {...}, \"options\": {...}}.",
            "required": ["businessLine"],
            "fields": list(INPUT_FIELDS),
            "note": CONTACT_NOTE,
            "options": {
                "retries": f"int 0-5, default {config.RETRIES}",
                "max_new_tokens": f"int 256-8192, default {config.MAX_NEW_TOKENS}",
                "seed": "int, makes a generation reproducible",
                "strict": "bool; unclean verification returns 422 instead of 200",
                "require_complete_inputs": "bool; reject rather than substitute a "
                                           "default for any missing financial "
                                           "field. Use for bank / NABARD "
                                           "submissions.",
                "include_instance": "bool; return the computed engine instance",
            },
        },
        "output": {
            "format": "JSON. The DPR is markdown; no PDF is rendered by this service.",
            "fields": {
                "markdown": "The DPR.",
                "verification": "ok=true means every figure in the markdown "
                                "matched the deterministic engine. number_issues "
                                "lists any that did not.",
                "meta": "Headline figures (total project cost, average DSCR, "
                        "year-1 PAT), token counts, timings, and which model "
                        "produced it.",
                "warnings": "Readable sentences for anything substituted or "
                            "not modelled.",
                "defaults_applied": "Machine-readable twin of `warnings`: every "
                                    "figure the form left blank, with the value, "
                                    "unit and source used. Empty means every "
                                    "figure came from the form.",
            },
            "json_schema": GenerateResponse.model_json_schema(),
        },
        "business_lines": business_lines(),
        "schemes": sorted(SCHEME_MAP),
        "limits": {
            "max_output_tokens": config.MAX_NEW_TOKENS,
            "typical_latency_s": "60-90",
            "max_latency_s": 240,
            "max_request_bytes": 1_048_576,
        },
        "determinism": (
            "Attempt 0 is greedy (temperature 0). Retries sample, and are only "
            "triggered when verification fails. Pass `options.seed` to make a "
            "sampled attempt reproducible. Financial figures do not depend on the "
            "model at all — they are computed before generation and re-checked "
            "after it."
        ),
    }
