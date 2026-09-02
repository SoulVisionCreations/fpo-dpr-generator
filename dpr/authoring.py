#!/usr/bin/env python3
"""
authoring.py - build the DPR authoring prompt from an engine instance.

This is the PRODUCTION / inference prompt builder. `build_full_prompt(instance)`
is pure, deterministic string construction from the instance — **no LLM is called
here**. It is used at inference (run_pipeline.py, eval_ft.py) to prompt the
finetuned model, and by the training tooling (teacher.py) to prompt the frontier
teacher. The teacher-LLM calls and the train/val dataset build live in teacher.py.

Option B: the model authors the WHOLE DPR (headings, §2 narrative, commentary) and
places engine content via {{TABLE:<id>}} placeholders; render.assemble then injects
the exact engine tables and verifies every number the model typed.

Usage (inspect a prompt):
  ./venv/bin/python authoring.py data/finetune/instances/rice_mill__g000.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dpr.render.assemble import section_catalog
from dpr.engine.financial_engine import compute

SYSTEM = """You are a senior agribusiness consultant authoring a complete Detailed \
Project Report (DPR) for an Indian Farmer Producer Organization (FPO), in the \
standard NIFTEM PM-FME format.

You write the ENTIRE document: the section headings, the full '## 2. Product, \
Market & Value Chain' narrative, and short interpretive commentary around the \
financial sections. You do NOT write the financial tables themselves - a \
deterministic engine owns every number. Where an engine table belongs, emit its \
placeholder on its own line, e.g. {{TABLE:5.7}}.

HARD RULES:
1. Reproduce the exact section structure given below, in order, with the given \
headings. Put each table's placeholder ({{TABLE:id}}) on its own line inside its \
section - exactly once, never omit one, never invent new ids.
2. Write '## 2.' as rich prose with ### sub-headings (Introduction, Uses & \
Benefits, Market Potential, Value Chain, Competition & Marketing, Risk & \
Mitigation, Employment & Socio-Economic Impact, Sustainability). No numbers there.
3. Write NO numbers anywhere in your prose - not in section 2, not in the \
commentary, not in section 3. EVERY figure lives in the injected tables. Interpret \
qualitatively only ("comfortable debt-service coverage", "modest capital outlay", \
"steadily improving profitability"). Do NOT cite the DSCR, project cost, capacity, \
percentages, or any other figure in words.
4. Never type a financial table yourself; always use the placeholder.
Output clean Markdown starting with the '#' title."""


def _computed(instance: dict) -> dict:
    if not instance.get("computed"):
        instance = json.loads(json.dumps(instance))
        compute(instance)
    return instance


def _idkey(sid: str):
    return tuple(int(p) for p in sid.split("."))


def build_full_prompt(instance: dict) -> tuple[str, str]:
    """Deterministically build the (system, user) authoring prompt for an instance:
    product facts + the exact section structure + the list of {{TABLE:id}} placeholders
    the model must place. No LLM involved."""
    inst = _computed(instance)
    n = inst.get("narrative", {})
    meta = inst["meta"]
    cat = section_catalog(inst)
    ids = sorted((i for i in cat if i not in ("2", "5")), key=_idkey)
    placeholders = "\n".join(f"  {{{{TABLE:{i}}}}} -- {cat[i]['heading']}" for i in ids)

    facts = "\n".join(x for x in [
        f"- Commodity / product: {meta.get('commodity')}",
        f"- Report title: {meta.get('report_title')}",
        (f"- Product note: {n.get('product_intro')}" if n.get("product_intro") else
         f"- Write the product description (Introduction, Uses & Benefits, Raw Material) "
         f"specifically for {meta.get('commodity')} from your own knowledge - accurate to "
         f"THIS product, qualitative only, no numbers."),
        f"- Raw-material note: {n.get('raw_material_note')}" if n.get("raw_material_note") else "",
        # the FPO's selected crop/commodity, when it isn't already the product name
        # (e.g. Nursery / Grading / Storage) — weave it naturally into the narrative
        (f"- Crop / commodity handled: {n.get('commodity_focus')} — reflect this in the narrative"
         if n.get("commodity_focus")
         and str(n.get("commodity_focus")).lower() not in str(meta.get("commodity", "")).lower() else ""),
        f"- Process: {' -> '.join(n.get('process_steps', []) or [])}",
        f"- FPO: {n.get('fpo_name') or 'the FPO'}, {n.get('location') or 'location'}",
    ] if x)

    structure = ("Document structure (write these headings in this order):\n"
                 "  # <report title>\n"
                 "  ## 1. Project Summary               -> {{TABLE:1}}\n"
                 "  ## 2. Product, Market & Value Chain -> your narrative (### sub-headings)\n"
                 "  ## 3. Basis & Presumptions          -> {{TABLE:3}}\n"
                 "  ## 4. Process Flow Chart            -> {{TABLE:4}}\n"
                 "  ## 5. Economics of the Project      -> each 5.x heading + its {{TABLE:5.x}}\n"
                 "  ## 6. Limitations of the Model DPR  -> {{TABLE:6}}")

    user = (f"Author the complete DPR.\n\nFACTS:\n{facts}\n\n"
            f"{structure}\n\nPLACE EACH OF THESE TABLES EXACTLY ONCE:\n{placeholders}\n\n"
            "Remember: NO numbers in your prose - all figures are in the injected tables. "
            "Write the full document now.")
    return SYSTEM, user


if __name__ == "__main__":
    inst = json.loads(Path(sys.argv[1]).read_text())
    sysp, usr = build_full_prompt(inst)
    print("=== SYSTEM ===\n" + sysp + "\n\n=== USER ===\n" + usr)
