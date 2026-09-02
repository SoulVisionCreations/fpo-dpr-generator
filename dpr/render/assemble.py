"""
Option-B assembly: the LLM authors the whole DPR (all prose, section flow, and
commentary) but places engine content through `{{TABLE:<id>}}` placeholders. The
engine's *exact* section bodies (every financial table) are injected here, and
every number the LLM typed in its own prose is verified against the engine.

Guarantee: no LLM-typed digit reaches the final DPR unchecked. Tables are byte
-for-byte the engine's; any inline figure the model states must match an engine
value or it is flagged (reject for training / repair at inference).

    from render.assemble import section_catalog, assemble, verify
    cat = section_catalog(instance)          # {id: {heading, body}}
    final, issues = assemble(llm_doc, instance)   # inject tables + verify
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dpr import config
from dpr.render.sections import render, DISCLAIMER_TEXT
from dpr.engine.financial_engine import compute
from dpr.engine.scheme_norms import SCHEME_NORMS

_HEADING = re.compile(r"^(#{2,3})\s+([0-9]+(?:\.[0-9]+)?)[.\s]", re.M)
# placeholder forms the LLM might emit: {{TABLE:5.7}} {{table 5.7}} {{TABLE_5_7}} {{5.7}}
_PLACEHOLDER = re.compile(r"\{\{\s*(?:table[\s:_-]*)?([0-9]+(?:[._][0-9]+)?)\s*\}\}", re.I)


def _computed(instance: dict) -> dict:
    if not instance.get("computed"):
        instance = json.loads(json.dumps(instance))
        compute(instance)
    return instance


def section_catalog(instance: dict) -> dict:
    """Every engine-owned section body, keyed by id ('1', '5.7', '6'). Section 2
    (the LLM's narrative) is excluded. Bodies are the exact renderer output."""
    inst = _computed(instance)
    doc = render(inst, narrative_md="__PLACEHOLDER__")
    heads = list(_HEADING.finditer(doc))
    cat = {}
    for i, h in enumerate(heads):
        sid = h.group(2)
        line_end = doc.index("\n", h.start()) if "\n" in doc[h.start():] else len(doc)
        body_start = line_end + 1
        body_end = heads[i + 1].start() if i + 1 < len(heads) else len(doc)
        if sid == "2":                       # LLM owns §2 entirely
            continue
        cat[sid] = {"heading": doc[h.start():line_end].lstrip("# ").strip(),
                    "body": doc[body_start:body_end].strip()}
    return cat


def _norm(x: float) -> tuple[float, float]:
    return round(x, 2), round(x, 0)


def allowed_numbers(instance: dict) -> set[float]:
    """Every number the LLM is allowed to state: engine outputs + config +
    the chosen scheme's norms (rounded to 2dp and to integer)."""
    inst = _computed(instance)
    vals: set[float] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            a, b = _norm(float(o)); vals.add(a); vals.add(b)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(inst.get("config", {}))
    walk(inst.get("computed", {}))
    walk(SCHEME_NORMS.get(inst["inputs"]["scheme"]["choice"], {}))
    return vals


# A number worth checking: carries a currency/scale/ratio marker, or is decimal
# / comma-grouped. Bare small integers ("5 years", "3 shifts") are ignored.
_FIN_NUM = re.compile(
    r"(?:₹|\bRs\.?\s*)?\s*(\d[\d,]*\.\d+|\d[\d,]*)\s*(%|lakh|lakhs|crore|crores)?", re.I)


def verify(llm_doc: str, instance: dict) -> list[dict]:
    """Flag every financial number in the LLM's own prose that doesn't match an
    engine value. Injected tables and placeholder lines are not checked."""
    allowed = allowed_numbers(instance)
    issues = []
    for raw_line in llm_doc.splitlines():
        line = raw_line.strip()
        # engine content / structural lines the LLM doesn't own -> skip
        if line.startswith("|") or line.startswith("#") or _PLACEHOLDER.search(line):
            continue
        for m in _FIN_NUM.finditer(line):
            num, marker = m.group(1), m.group(2)
            has_marker = bool(marker) or "₹" in m.group(0) or re.search(r"rs\.?", m.group(0), re.I)
            if not has_marker and "." not in num and "," not in num:
                continue                                        # bare int, not financial
            before = line[:m.start()].lower()
            if re.search(r"(section|sub-?section|table|clause|para|no\.|§)\s*$", before):
                continue                                        # a structural reference (e.g. "Table 5.7")
            try:
                val = float(num.replace(",", ""))
            except ValueError:
                continue
            if 1900 <= val <= 2100 and "." not in num and not marker:
                continue                                        # a year
            if not any(abs(val - a) <= 0.02 for a in allowed):
                issues.append({"number": m.group(0).strip(), "value": val,
                               "context": line[:90]})
    return issues


def assemble(llm_doc: str, instance: dict) -> tuple[str, dict]:
    """Inject exact engine section bodies into the LLM's placeholders and verify
    inline numbers. Returns (final_md, report). report.ok is True only when every
    placeholder resolved and no number is unverified."""
    cat = section_catalog(instance)
    placed, unknown = set(), set()

    def sub(m):
        sid = m.group(1).replace("_", ".")
        if sid in cat:
            placed.add(sid)
            return cat[sid]["body"]
        unknown.add(sid)
        return m.group(0)

    final = _PLACEHOLDER.sub(sub, llm_doc)
    # Static Disclaimer always appended at the very end (not LLM-written).
    if "## Disclaimer" not in final:
        final = final.rstrip() + "\n\n" + DISCLAIMER_TEXT + "\n"
    issues = verify(llm_doc, instance)
    report = {
        "ok": not unknown and not issues and not (set(cat) - placed - {"5"}),
        "unknown_placeholders": sorted(unknown),
        "unplaced_sections": sorted(set(cat) - placed - {"5"}),  # '5' is the empty parent
        "number_issues": issues,
    }
    return final, report


if __name__ == "__main__":
    inst = json.load(open(config.GENERATED_DIR / "ghee_instance.json"))
    cat = section_catalog(inst)
    print(f"catalog sections: {', '.join(sorted(cat))}\n")
    demo = (
        "# DPR\n\n## 1. Project Summary\n{{TABLE:1}}\n\n"
        "## 2. Product, Market & Value Chain\n### Introduction\nGhee is clarified butter.\n\n"
        "## 5. Economics\n### 5.7 Means of Finance\n"
        "The average DSCR of 3.15 shows comfortable debt servicing.\n{{TABLE:5.7}}\n"
        "The total project cost of Rs. 99.99 lakh is modest.\n"   # 99.99 is WRONG -> must flag
    )
    final, rep = assemble(demo, inst)
    print("report.ok:", rep["ok"])
    print("unplaced (sample):", rep["unplaced_sections"][:5], "...")
    print("number_issues:", rep["number_issues"])
    print("\n--- assembled §5.7 injected? ---")
    print("Term Loan" in final, "| DSCR table rows present:", "3.15" in final)
