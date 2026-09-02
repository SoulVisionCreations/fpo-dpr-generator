#!/usr/bin/env python3
"""
score_dpr.py - heuristic quality scorer for the LLM-written section 2 of a DPR.

Only "## 2. Product, Market & Value Chain" is model-written; every other section
is deterministic engine output. So we score ONLY section 2, encoding the criteria
used by hand in the V3-V6 model comparisons:

  process_grounding (30) - does the prose reflect the real product/process?
  subsection_coverage (20) - are all 8 expected sub-sections present?
  no_leaked_figures (20) - the engine owns the numbers; section 2 must have none.
  conciseness (15) - tight, DPR-style prose (penalise verbosity + thinness).
  business_model_fidelity (10) - service vs batch vs processing framed correctly.
  low_repetition (5) - little n-gram repetition.

Total 0-100. When the source instance can't be found, the two instance-dependent
dimensions (grounding, business-model) are dropped and the rest re-weighted to 100.

Usage:
    ./venv/bin/python score_dpr.py output/DPR_ghee__gpt-oss-20b_20260716.md
    ./venv/bin/python score_dpr.py output/DPR_ghee__*.md --json
    ./venv/bin/python score_dpr.py --dir output               # batch, ranked table
    ./venv/bin/python score_dpr.py --dir output --by-model    # per-model averages
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from dpr import config
GENERATED = config.GENERATED_DIR

WEIGHTS = {  # must sum to 100
    "process_grounding": 30,
    "subsection_coverage": 20,
    "no_leaked_figures": 20,
    "conciseness": 15,
    "business_model_fidelity": 10,
    "low_repetition": 5,
}
INSTANCE_DEPENDENT = ("process_grounding", "business_model_fidelity")

# The 8 expected section-2 sub-sections, each as a set of heading keywords
# (models phrase them differently: "Value Chain Analysis", "Value Chain
# Integration", "Customer Value Proposition" all count as value chain).
SUBSECTIONS = {
    "introduction": ("introduction", "product overview", "the product"),
    "uses_benefits": ("use", "uses", "benefit", "application"),
    "market": ("market", "demand", "potential"),
    "value_chain": ("value chain", "value-chain", "value proposition"),
    "competition_marketing": ("competition", "competitor", "marketing", "strategy"),
    "risk": ("risk", "mitigation"),
    "employment": ("employment", "socio", "economic impact", "social impact", "livelihood"),
    "sustainability": ("sustainab",),
}

# --- leaked-figure detectors -------------------------------------------------
# A leaked figure is an *amount*, so every alternative requires a digit —
# the bare word "rupee(s)" in prose ("better rupee realization") is not a leak.
_CURRENCY = re.compile(r"(₹\s*\d|\bRs\.?\s*\d|\b\d[\d,]*\.?\d*\s*(?:lakh|lakhs|crore|crores)\b|"
                       r"\brupees?\s*\d|\d[\d,]*\s*rupees?\b)", re.I)
_METRIC = re.compile(r"\b(DSCR|IRR|NPV|payback|break[\s-]?even|debt[\s-]?service|"
                     r"term loan|total project cost|working capital|net profit|"
                     r"profit after tax|\bPAT\b|means of finance)\b", re.I)
_PCT = re.compile(r"\d+(?:\.\d+)?\s*%")
_PCT_FIN_CTX = ("margin", "subsid", "return", "profit", "interest", "escalat",
                "utilis", "utiliz", "capacity", "irr", "dscr", "roi", "depreciat",
                "tax", "equity", "loan")

_STOP = {
    "and", "the", "for", "from", "into", "with", "this", "that", "than", "then",
    "are", "was", "will", "per", "its", "our", "their", "which", "such", "also",
    "procurement", "storage", "packing", "packaging", "dispatch", "quality",
    "process", "processing", "unit", "material", "materials", "product", "products",
    "management", "required", "requirement",
}


def extract_section2(md: str) -> str:
    """Return the text of '## 2. ...' up to the next top-level '## ' heading."""
    m = re.search(r"^##\s*2\.\s.*$", md, re.M)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"^##\s*\d", md[start:], re.M)
    return md[start: start + nxt.start()] if nxt else md[start:]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())


def score_coverage(s2: str) -> tuple[float, dict]:
    headings = " || ".join(re.findall(r"^###\s+(.+)$", s2, re.M)).lower()
    found = {name: any(k in headings for k in kws) for name, kws in SUBSECTIONS.items()}
    n = sum(found.values())
    return n / len(SUBSECTIONS), {"found": n, "of": len(SUBSECTIONS),
                                  "missing": [k for k, v in found.items() if not v]}


def score_leaks(s2: str) -> tuple[float, dict]:
    hard = list(_CURRENCY.findall(s2)) + _METRIC.findall(s2)
    soft = 0
    hard_pct = 0
    for m in _PCT.finditer(s2):
        window = s2[max(0, m.start() - 45): m.end() + 45].lower()
        if any(w in window for w in _PCT_FIN_CTX):
            hard_pct += 1
        else:
            soft += 1
    n_hard = len(hard) + hard_pct
    score = max(0.0, 1.0 - 0.34 * n_hard - 0.05 * soft)
    return score, {"hard_leaks": n_hard, "soft_pct": soft}


def _vocab(instance: dict) -> set[str]:
    n = instance.get("narrative", {})
    parts = list(n.get("process_steps", []) or [])
    parts.append(instance.get("meta", {}).get("commodity", ""))
    parts.append(n.get("raw_material_note", "") or "")
    parts.append(n.get("product_intro", "") or "")
    toks = {w for p in parts for w in _words(p) if len(w) >= 4 and w not in _STOP}
    return toks


def score_grounding(s2: str, instance: dict) -> tuple[float, dict]:
    vocab = _vocab(instance)
    if not vocab:
        return 0.0, {"matched": 0, "target": 0}
    s2l = s2.lower()
    matched = {t for t in vocab if t in s2l}
    target = min(10, max(4, round(0.6 * len(vocab))))
    return min(1.0, len(matched) / target), {"matched": len(matched), "target": target,
                                             "vocab": len(vocab)}


def _mode(instance: dict) -> str:
    cap = instance.get("inputs", {}).get("capacity", {})
    if instance.get("meta", {}).get("revenue_model") == "service" or "annual_service_units" in cap:
        return "service"
    if "annual_input_units" in cap:
        return "batch"
    return "processing"


def score_business_model(s2: str, instance: dict) -> tuple[float, dict]:
    s2l = s2.lower()
    mode = _mode(instance)
    if mode == "service":
        pos = ("rent", "hire", "rental", "lease", "tariff", "per hour", "per tonne",
               "per acre", "machine-hour", "occupancy", "storage", "service")
        # Only penalise framing that casts THIS unit as a manufacturer; generic
        # mentions ("parts manufacturing" as ancillary employment) don't count.
        mfg = ("manufacturing unit", "manufacturing facility", "manufacturing plant",
               "manufacturing process", "manufacture the", "raw material into",
               "convert into product", "production line", "processed into a")
        score = 1.0 if any(w in s2l for w in pos) else 0.5
        if any(w in s2l for w in mfg):
            score -= 0.3
        return max(0.0, score), {"mode": mode}
    if mode == "batch":
        kw = ("harden", "plantlet", "sapling", "acclimatis", "acclimatiz", "nursery",
              "polyhouse", "shade net", "shade-net", "cocopeat", "mortality", "seedling")
        return (1.0 if sum(w in s2l for w in kw) >= 2 else 0.5), {"mode": mode}
    commodity = instance.get("meta", {}).get("commodity", "").split()
    hit = any(c.lower() in s2l for c in commodity if len(c) >= 3)
    return (1.0 if hit else 0.7), {"mode": mode}


def score_conciseness(s2: str) -> tuple[float, dict]:
    w = len(_words(s2))
    if 700 <= w <= 1300:
        s = 1.0
    elif w < 700:
        s = max(0.35, 0.5 + 0.5 * (w - 400) / 300)     # 400->0.5, 700->1.0
    else:
        s = max(0.30, 1.0 - (w - 1300) / 700.0)        # 1300->1.0, 2000->0.0
    return s, {"words": w}


def score_repetition(s2: str) -> tuple[float, dict]:
    w = _words(s2)
    if len(w) < 50:
        return 1.0, {"trigram_uniqueness": 1.0}
    tris = list(zip(w, w[1:], w[2:]))
    uniq = len(set(tris)) / len(tris)
    s = max(0.0, min(1.0, (uniq - 0.55) / 0.40))       # 0.55->0, 0.95->1
    return s, {"trigram_uniqueness": round(uniq, 3)}


def find_instance(dpr_path: Path) -> dict | None:
    m = re.match(r"DPR_(.+?)__", dpr_path.name)
    if not m:
        return None
    p = GENERATED / f"{m.group(1)}_instance.json"
    return json.loads(p.read_text()) if p.exists() else None


def score_file(dpr_path: Path) -> dict:
    md = dpr_path.read_text()
    s2 = extract_section2(md)
    inst = find_instance(dpr_path)
    dims, detail = {}, {}
    dims["subsection_coverage"], detail["subsection_coverage"] = score_coverage(s2)
    dims["no_leaked_figures"], detail["no_leaked_figures"] = score_leaks(s2)
    dims["conciseness"], detail["conciseness"] = score_conciseness(s2)
    dims["low_repetition"], detail["low_repetition"] = score_repetition(s2)
    if inst is not None:
        dims["process_grounding"], detail["process_grounding"] = score_grounding(s2, inst)
        dims["business_model_fidelity"], detail["business_model_fidelity"] = score_business_model(s2, inst)

    active = {k: v for k, v in WEIGHTS.items() if k in dims}
    wsum = sum(active.values())
    total = sum(dims[k] * active[k] for k in dims) / wsum * 100.0

    m = re.match(r"DPR_(.+?)__(.+?)_\d{8}", dpr_path.name)
    return {
        "file": dpr_path.name,
        "line": m.group(1) if m else dpr_path.stem,
        "model": m.group(2) if m else "?",
        "instance_found": inst is not None,
        "total": round(total, 1),
        "subscores": {k: round(dims[k], 3) for k in dims},
        "detail": detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Heuristic §2 quality scorer for DPRs.")
    ap.add_argument("files", nargs="*", help="DPR .md files")
    ap.add_argument("--dir", help="score every DPR_*.md in this directory")
    ap.add_argument("--glob", default="DPR_*__*.md", help="glob within --dir")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--by-model", action="store_true", help="print per-model averages")
    args = ap.parse_args()

    paths = [Path(f) for f in args.files]
    if args.dir:
        paths += [Path(p) for p in sorted(glob.glob(str(Path(args.dir) / args.glob)))]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no DPR files given", file=sys.stderr)
        return 1

    results = [score_file(p) for p in paths]
    results.sort(key=lambda r: r["total"], reverse=True)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\n{'DPR (line / model)':52} {'score':>6}  cov leak grnd conc bmf  rep")
    print("-" * 92)
    for r in results:
        s = r["subscores"]
        g = lambda k: f"{s[k]*100:3.0f}" if k in s else "  -"
        print(f"{r['line'][:30]:30} {r['model'][:20]:20} {r['total']:6.1f}  "
              f"{g('subsection_coverage')} {g('no_leaked_figures')} {g('process_grounding')} "
              f"{g('conciseness')} {g('business_model_fidelity')} {g('low_repetition')}")

    if args.by_model:
        by = {}
        for r in results:
            by.setdefault(r["model"], []).append(r["total"])
        print("\nPer-model average (n):")
        for mdl, xs in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(f"  {mdl:22} {sum(xs)/len(xs):5.1f}  (n={len(xs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
