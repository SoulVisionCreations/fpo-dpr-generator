#!/usr/bin/env python3
"""
gen_dataset.py - generate diverse, VIABLE synthetic DPR instances for building
fine-tuning data (SFT distillation of section-2 narrative).

Section 2 (the only model-written part) depends mainly on business line /
commodity / scale / location / buyers, NOT on the exact rupee figures. So we
perturb those FPO-specific knobs on each validated base payload and keep only
variants that still compute cleanly (means of finance balances, PAT >= 0,
avg DSCR >= 1.5). Prices/scheme/land stay near the calibrated base so viability
holds and §2 stays grounded.

Output:
  data/finetune/instances/<line>__<id>.json   (engine instances, computed={})
  data/finetune/manifest.jsonl                 (one row per instance + metrics)

Next step (needs a frontier API key): teacher.py turns each into a
build_prompt() and calls the teacher model to produce the gold §2.

Usage:
  ./venv/bin/python gen_dataset.py --per-commodity 24 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

from dpr import config
from dpr.form_to_instance import convert
from dpr.engine.financial_engine import compute

PAYLOADS = config.PAYLOADS_DIR
OUT = config.FINETUNE_DIR

NAME_PREFIX = ["Shri", "Jai", "Maa", "Sri", "Anna", "Kisan", "Sahyadri", "Godavari",
               "Narmada", "Kaveri", "Bhoomi", "Surya", "Ganga", "Vindhya", "Aravalli",
               "Satpura", "Malwa", "Konkan", "Deccan", "Terai"]
LOCATIONS = [
    {"village": "Warud", "block": "Warud", "district": "Amravati", "state": "Maharashtra"},
    {"village": "Sirsi", "block": "Sirsi", "district": "Uttara Kannada", "state": "Karnataka"},
    {"village": "Palladam", "block": "Tiruppur", "district": "Tiruppur", "state": "Tamil Nadu"},
    {"village": "Barabanki", "block": "Nawabganj", "district": "Barabanki", "state": "Uttar Pradesh"},
    {"village": "Anand", "block": "Anand", "district": "Anand", "state": "Gujarat"},
    {"village": "Karimnagar", "block": "Karimnagar", "district": "Karimnagar", "state": "Telangana"},
    {"village": "Sangrur", "block": "Sangrur", "district": "Sangrur", "state": "Punjab"},
    {"village": "Kota", "block": "Ladpura", "district": "Kota", "state": "Rajasthan"},
    {"village": "Nadia", "block": "Krishnanagar", "district": "Nadia", "state": "West Bengal"},
    {"village": "Guntur", "block": "Guntur", "district": "Guntur", "state": "Andhra Pradesh"},
    {"village": "Dewas", "block": "Dewas", "district": "Dewas", "state": "Madhya Pradesh"},
    {"village": "Hassan", "block": "Hassan", "district": "Hassan", "state": "Karnataka"},
]


def perturb(payload: dict, rng: random.Random) -> dict:
    p = deepcopy(payload)
    s0, s1, costs = p["stage0_basic_info"], p["stage1_land_business"], p["project_costs_and_prices"]
    crop = (s0.get("top_crops") or ["Agri"])[0]
    s0["fpo_name"] = f"{rng.choice(NAME_PREFIX)} {crop} Farmer Producer Company Ltd"
    s0["location"] = dict(rng.choice(LOCATIONS))
    s0["total_members"] = rng.randint(300, 6000)
    s0["annual_turnover_inr"] = s0["total_members"] * rng.randint(3000, 12000)
    s0["year_of_registration"] = rng.randint(2015, 2023)

    cap = s1["target_capacity"]                          # scale capacity (margin-neutral)
    for k in ("input_per_day", "annual_units"):
        if k in cap:
            cap[k] = round(cap[k] * rng.uniform(0.7, 1.4), 3)
    s1["expected_direct_jobs"] = rng.randint(4, 20)

    f = rng.uniform(0.96, 1.06)                          # gentle price jitter (keep margin)
    costs["product_selling_price_inr_per_unit"] = round(costs["product_selling_price_inr_per_unit"] * f, 2)
    rm = costs["raw_material"]
    if rm["costing_method"] == "per_unit":
        rm["value"] = round(rm["value"] * rng.uniform(0.96, 1.05), 2)

    # own funds >= ~10% of machinery so the promoter-contribution check passes
    s1["own_funds_committed_inr"] = int(costs["machinery_total_lakh"] * 1e5 * rng.uniform(0.10, 0.22))
    if s1.get("anticipated_buyers"):                     # light market-side variety
        b = list(s1["anticipated_buyers"]); rng.shuffle(b)
        s1["anticipated_buyers"] = b[: max(2, len(b) - rng.randint(0, 1))]
    return p


def viability(inst: dict):
    c = compute(deepcopy(inst))["computed"]
    mf, pc, pr, d = c["means_of_finance"], c["project_cost"], c["profitability"], c["dscr"]
    ok = (abs(mf["total"] - pc["total_project_cost"]) <= 0.05
          and all(x >= 0 for x in pr["pat"]) and d["average"] >= 1.5)
    return ok, {"tpc": pc["total_project_cost"], "avg_dscr": d["average"], "pat_y1": pr["pat"][0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-commodity", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tries", type=int, default=6, help="attempts per wanted instance")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    (OUT / "instances").mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"
    rows, kept, tried = [], 0, 0

    for pf in sorted(PAYLOADS.glob("*_payload.json")):
        payload = json.loads(pf.read_text())
        base = pf.stem.replace("_payload", "")
        n = 0
        for i in range(args.per_commodity * args.max_tries):
            if n >= args.per_commodity:
                break
            tried += 1
            try:
                inst = convert(perturb(payload, rng))
                ok, m = viability(inst)
            except Exception:
                continue
            if not ok:
                continue
            iid = f"{base}__{n:03d}"
            (OUT / "instances" / f"{iid}.json").write_text(json.dumps(inst, ensure_ascii=False))
            rows.append({"id": iid, "base": base,
                         "business_line": inst["meta"].get("commodity"),
                         "revenue_model": inst["meta"].get("revenue_model"),
                         **m})
            n += 1
            kept += 1
        print(f"  {base:34} kept {n}/{args.per_commodity}")

    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\n[ok] {kept} viable instances (from {tried} tries) -> {OUT/'instances'}")
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
