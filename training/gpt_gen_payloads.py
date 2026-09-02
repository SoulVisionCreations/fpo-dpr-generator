#!/usr/bin/env python3
"""
gpt_gen_payloads.py - generate diverse FPO form payloads, convert them to engine
instances, and keep only the VIABLE ones for the fine-tuning pool.

GPT invents the SCENARIO (the FPO-specific form payload); the engine + templates
still own the numbers and guarantee viability. Any payload that doesn't compute to
a viable DPR (PAT >= 0, means-of-finance balanced, avg DSCR >= target) is discarded.
This gives GPT-level diversity without trusting GPT with the math.

  payload (GPT)  ->  form_to_instance  ->  engine.compute  ->  viability filter  ->  instance

Modes:
  --dry-run              deterministic perturbation of the base payloads (NO API);
                         validates the whole loop end-to-end without a key.
  --provider openai      GPT authors the payloads (needs OPENAI_API_KEY).
  --provider anthropic   Claude authors the payloads (needs ANTHROPIC_API_KEY).

Output: viable instances -> data/finetune/instances/<base>__g<NNN>.json
        (teacher.py then authors DPRs for the whole pool)

Usage:
  ./venv/bin/python gpt_gen_payloads.py --dry-run --per-line 4
  OPENAI_API_KEY=sk-... ./venv/bin/python gpt_gen_payloads.py --provider openai --model gpt-4o --per-line 40
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

from dpr import config
from dpr.form_to_instance import convert
from training.gen_dataset import perturb, viability


def _load_dotenv():
    """Load BASE/.env into os.environ (same convention as author_v3.sh) so the
    provider API key is picked up when this script is run directly. Existing env
    vars win, so an intentional export still overrides .env."""
    import os
    envf = config.REPO_ROOT / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

PAYLOADS = config.PAYLOADS_DIR
TEMPLATES = config.TEMPLATES_DIR
OUT = config.FINETUNE_DIR / "instances"

PAYLOAD_SYSTEM = (
    "You generate realistic synthetic INPUT DATA for Indian Farmer Producer "
    "Organization (FPO) project reports. You output ONLY valid JSON — a single JSON "
    "array of payload objects — with no prose, no explanation, and no code fences.")


def crop_hint(base_stem: str, payload: dict) -> str:
    tc = payload["stage0_basic_info"].get("top_crops") or []
    bl = payload["stage0_basic_info"]["business_line"]
    if bl == "rice_dal_flour_mill" or "mill" in base_stem:
        return "a milling grain whose FIRST top-crop routes the mill (paddy->rice, wheat->flour, gram/pulse->dal)"
    # Commodity-variable processing lines: the product is derived from the FIRST
    # top-crop, so VARY it widely across payloads (this is what gives the v4 model
    # its oilseed / fruit / spice breadth instead of a mustard-/amla-only bias).
    if bl == "oil_extraction_unit":
        return ("make the FIRST top-crop a DIFFERENT oilseed on each payload — "
                "mustard, groundnut, sesame (til), sunflower, soybean, niger, coconut, safflower")
    if bl == "processing_unit_fruit_juice":
        return ("make the FIRST top-crop a DIFFERENT fruit on each payload — "
                "amla, mango, guava, pineapple, orange, pomegranate, litchi, grape, jamun, bael")
    if bl in ("processing_unit_spices", "processing_unit"):
        return ("make the FIRST top-crop a DIFFERENT spice on each payload — "
                "turmeric, chilli, coriander, cumin, ginger, black pepper, fenugreek, fennel")
    return f"crops consistent with {tc[:2] or 'this business'}"


REGIONS = ["north India (Punjab/Haryana/UP)", "south India (TN/Karnataka/AP/Telangana)",
           "west India (Maharashtra/Gujarat/MP)", "east India (WB/Bihar/Odisha/Jharkhand)",
           "central & tribal belt (Chhattisgarh/MP/Vidarbha)", "north-east & hill states"]


def build_payload_prompt(base_stem: str, payload: dict, n: int, variant: int) -> tuple[str, str]:
    bl = payload["stage0_basic_info"]["business_line"]
    example = json.dumps(payload, ensure_ascii=False, indent=2)
    region = REGIONS[variant % len(REGIONS)]
    scale = ["small (0.5x-0.9x the example)", "mid (0.9x-1.4x the example)",
             "large (1.4x-2x the example)"][variant % 3]
    user = (
        f"Example form payload for a '{bl}' FPO:\n{example}\n\n"
        f"Generate {n} DIFFERENT, realistic Indian FPO payloads for the SAME "
        f"business_line ('{bl}'). This is batch #{variant}: locate these FPOs in "
        f"{region} (use REAL villages/blocks/districts/states from that region) and "
        f"skew the operation size toward {scale}. Make every FPO distinct from the "
        f"others — vary fpo_name, location, year_of_registration, total_members, "
        f"annual_turnover_inr, top_crops ({crop_hint(base_stem, payload)}), "
        f"target_capacity, scheme_choice, land_status, own_funds_committed_inr, "
        f"machinery_total_lakh, civil_construction_lakh, and prices. Keep the product "
        f"SALE price above the raw-material cost with a realistic margin (as in the "
        f"example). Use the EXACT same JSON structure and field names as the example. "
        f"Output ONLY a JSON array of {n} payload objects.")
    return PAYLOAD_SYSTEM, user


def parse_payloads(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("[")
    if start < 0:
        return []
    obj = json.loads(text[start: text.rfind("]") + 1])
    return obj if isinstance(obj, list) else [obj]


def gen_chunk(base_stem: str, payload: dict, n: int, variant: int, args) -> list[dict]:
    """One API call producing up to n payloads (one 'batch'/variant)."""
    from training.teacher import call_teacher
    sysp, usr = build_payload_prompt(base_stem, payload, n, variant)
    try:
        return parse_payloads(call_teacher(args.provider, args.model, sysp, usr))
    except Exception as e:
        print(f"  [warn] {base_stem} batch#{variant}: payload gen failed "
              f"({type(e).__name__}: {str(e)[:80]})")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="deterministic perturbation, no API")
    ap.add_argument("--provider", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--per-line", type=int, default=4, help="payloads requested per base payload")
    ap.add_argument("--chunk", type=int, default=8,
                    help="payloads per API call (large --per-line is split into batches so a "
                         "reasoning model doesn't truncate)")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel API calls (--provider)")
    ap.add_argument("--target", type=int,
                    help="top-up mode: keep generating per commodity until this many VIABLE "
                         "instances exist (counts + appends to what's already in instances/, "
                         "auto-balancing across the variable viability yield)")
    ap.add_argument("--max-rounds", type=int, default=8,
                    help="safety cap on top-up rounds (low-yield commodities may fall short)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.dry_run and not args.provider:
        print("give --dry-run or --provider", file=sys.stderr)
        return 1
    if args.provider:
        _load_dotenv()

    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    bases = [(pf.stem.replace("_payload", ""), json.loads(pf.read_text()))
             for pf in sorted(PAYLOADS.glob("*_payload.json"))]

    # DRY-RUN: deterministic perturbation, no API, sequential (fast).
    if args.dry_run:
        kept = tried = 0
        for base, payload in bases:
            n = 0
            for _ in range(args.per_line):
                tried += 1
                try:
                    inst = convert(perturb(payload, rng)); ok, _ = viability(inst)
                except Exception:
                    continue
                if not ok:
                    continue
                (OUT / f"{base}__g{n:03d}.json").write_text(json.dumps(inst, ensure_ascii=False))
                n += 1; kept += 1
            print(f"  {base:28} kept {n}/{args.per_line}")
        print(f"\n[ok] {kept} viable instances (from {tried} payloads) -> {OUT}")
        return 0

    import concurrent.futures as cf

    def existing_count(base: str) -> int:
        return len(list(OUT.glob(f"{base}__g*.json")))

    def next_index(base: str) -> int:
        idxs = [int(m.group(1)) for f in OUT.glob(f"{base}__g*.json")
                if (m := re.search(r"__g(\d+)\.json$", f.name))]
        return max(idxs) + 1 if idxs else 0

    def keep_viable(base: str, payloads: list[dict], idx: int, room: int) -> tuple[int, int]:
        """Convert+filter payloads, write up to `room` viable, return (written, next_idx)."""
        w = 0
        for p in payloads:
            if w >= room:
                break
            try:
                inst = convert(p); ok, _ = viability(inst)
            except Exception:
                continue
            if not ok:
                continue
            (OUT / f"{base}__g{idx:03d}.json").write_text(json.dumps(inst, ensure_ascii=False))
            idx += 1; w += 1
        return w, idx

    # TARGET (top-up) MODE: fire proportional chunks per under-target commodity each
    # round, append viable up to target, repeat until every commodity is filled (or the
    # round cap is hit). Auto-balances across the wildly variable per-commodity yield.
    if args.target:
        have = {b: existing_count(b) for b, _ in bases}
        idx = {b: next_index(b) for b, _ in bases}
        yield_est = {b: 0.3 for b, _ in bases}  # adaptive; updated from observed
        print(f"[top-up] target={args.target}/commodity | have now: "
              f"{sum(have.values())} across {len(bases)} commodities "
              f"(chunk={args.chunk}, concurrency={args.concurrency}, max_rounds={args.max_rounds})")
        for rnd in range(1, args.max_rounds + 1):
            under = [(b, p) for b, p in bases if have[b] < args.target]
            if not under:
                break
            # per commodity: enough chunks to cover the deficit at the current yield estimate
            jobs = []  # (base, payload, chunk_size, variant)
            for b, p in under:
                deficit = args.target - have[b]
                need_payloads = deficit / max(yield_est[b], 0.08)
                nch = max(1, min(6, -(-int(need_payloads) // args.chunk)))  # ceil, capped 6/round
                for c in range(nch):
                    jobs.append((b, p, args.chunk, rnd * 100 + c))
            print(f"  round {rnd}: {len(under)} under target, firing {len(jobs)} chunk-calls")
            gen = {b: [] for b, _ in bases}
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(gen_chunk, b, p, cs, v, args): b for (b, p, cs, v) in jobs}
                for fut in cf.as_completed(futs):
                    b = futs[fut]
                    try:
                        gen[b].extend(fut.result())
                    except Exception as e:
                        print(f"    [warn] {b}: {type(e).__name__}: {str(e)[:70]}")
            for b, _ in bases:
                if not gen[b]:
                    continue
                room = args.target - have[b]
                w, idx[b] = keep_viable(b, gen[b], idx[b], room)
                have[b] += w
                obs = w / len(gen[b])
                yield_est[b] = 0.5 * yield_est[b] + 0.5 * obs  # smooth toward observed
            done = sum(1 for b, _ in bases if have[b] >= args.target)
            print(f"    after round {rnd}: {sum(have.values())} total, "
                  f"{done}/{len(bases)} commodities at target")
        print("\n[ok] top-up complete. per-commodity viable counts:")
        for b, _ in bases:
            flag = "" if have[b] >= args.target else f"  (short of {args.target})"
            print(f"  {b:28} {have[b]}{flag}")
        print(f"\n[ok] {sum(have.values())} viable instances total -> {OUT}")
        print("Next: ./venv/bin/python authoring.py --provider openai --model "
              f"{args.model} --concurrency {args.concurrency}   then   --collect")
        return 0

    # PROVIDER: split each base's --per-line into chunks, fire all chunks across all
    # bases in parallel, then convert+viability-filter+write (sequential numbering).
    tasks = []  # (base, payload, chunk_size, variant)
    for base, payload in bases:
        remaining, v = args.per_line, 0
        while remaining > 0:
            tasks.append((base, payload, min(args.chunk, remaining), v))
            remaining -= args.chunk; v += 1
    print(f"[payloads] {len(bases)} commodities x {args.per_line} = "
          f"{len(tasks)} API calls (chunk={args.chunk}, concurrency={args.concurrency}) "
          f"| model={args.model}")

    by_base: dict[str, list[dict]] = {b: [] for b, _ in bases}
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(gen_chunk, b, p, cs, v, args): b for (b, p, cs, v) in tasks}
        for fut in cf.as_completed(futs):
            base = futs[fut]
            try:
                by_base[base].extend(fut.result())
            except Exception as e:
                print(f"  [warn] {base}: {type(e).__name__}: {str(e)[:80]}")

    kept = tried = 0
    for base, _ in bases:
        n = 0
        for p in by_base[base]:
            tried += 1
            try:
                inst = convert(p); ok, _ = viability(inst)
            except Exception:
                continue
            if not ok:
                continue
            (OUT / f"{base}__g{n:03d}.json").write_text(json.dumps(inst, ensure_ascii=False))
            n += 1; kept += 1
        print(f"  {base:28} kept {n} (from {len(by_base[base])} generated)")
    print(f"\n[ok] {kept} viable instances (from {tried} payloads) -> {OUT}")
    print("Next: ./venv/bin/python authoring.py --provider openai --model "
          f"{args.model} --concurrency {args.concurrency}   then   --collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
