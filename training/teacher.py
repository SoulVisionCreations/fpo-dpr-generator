#!/usr/bin/env python3
"""
teacher.py - training-data generation via a frontier teacher LLM.

TRAINING ONLY (never used at inference/production). It takes the deterministic
prompt from authoring.build_full_prompt, calls a frontier teacher (gpt-5 / claude)
to author gold DPRs, keeps only those that assemble + verify cleanly, and writes
train/val.jsonl for finetuning. The student model learns to imitate these; after
finetuning the teacher is retired.

  --dry-run       : deterministic stand-in teacher (no API) — proves the loop
  --provider      : call the real frontier teacher, author into teacher_out/
  --collect       : assemble+verify teacher_out/*.md -> train/val.jsonl
  --dump-prompts  : write prompts/ for a manual (Claude Code / claude.ai) teacher
  --show-prompt   : print one instance's prompt and exit

Usage:
  ./venv/bin/python teacher.py --dry-run --limit 20
  OPENAI_API_KEY=... ./venv/bin/python teacher.py --provider openai --model gpt-5 --concurrency 8
  ./venv/bin/python teacher.py --collect --val-frac 0.28
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

from dpr import config
from dpr.authoring import build_full_prompt, _computed, _idkey
from dpr.render.assemble import section_catalog, assemble

FT = config.FINETUNE_DIR
PROMPTS = FT / "prompts"
TEACHER_OUT = FT / "teacher_out"


# --- deterministic stand-in teacher (dry-run only; no API) -------------------
def stub_teacher(instance: dict) -> str:
    inst = _computed(instance)
    n, meta = inst.get("narrative", {}), inst["meta"]
    cat = section_catalog(inst)
    d = inst["computed"]["dscr"]["average"]
    prod = n.get("product_intro") or f"This project concerns {meta.get('commodity')}."
    rm = n.get("raw_material_note") or "Raw material is sourced locally."
    subs = ["Introduction to the Product", "Uses & Benefits", "Market Potential & Demand",
            "Value Chain", "Competition & Marketing Strategy", "Risk Factors & Mitigation",
            "Employment & Socio-Economic Impact", "Sustainability"]
    body = []
    for i, s in enumerate(subs):
        txt = prod if i == 0 else rm if i == 3 else \
            f"This section discusses {s.lower()} for the enterprise in qualitative terms."
        body.append(f"### {s}\n\n{txt}")
    s2 = "## 2. Product, Market & Value Chain\n\n" + "\n\n".join(body)

    out = [f"# {meta.get('report_title', 'DETAILED PROJECT REPORT')}",
           "## 1. Project Summary", "{{TABLE:1}}", s2,
           "## 3. Basis & Presumptions", "{{TABLE:3}}",
           "## 4. Process Flow Chart", "{{TABLE:4}}",
           "## 5. Economics of the Project",
           f"The projected average debt-service coverage ratio of {d:,.2f} indicates comfortable debt servicing."]
    for sid in sorted((i for i in cat if i.startswith("5.")), key=_idkey):
        out += [f"### {cat[sid]['heading']}", f"{{{{TABLE:{sid}}}}}"]
    out += ["## 6. Limitations of the Model DPR", "{{TABLE:6}}"]
    return "\n\n".join(out)


# --- real frontier teacher (stdlib urllib; only used with --provider) --------
def call_teacher(provider: str, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": model, "max_tokens": 8000, "system": system,
                             "messages": [{"role": "user", "content": user}]}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)["content"][0]["text"]
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
        body = {"model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        # reasoning models (gpt-5, o-series) use max_completion_tokens (which also
        # covers hidden reasoning tokens -> needs a big budget) and only support the
        # default temperature; classic chat models use max_tokens.
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["max_completion_tokens"] = 16000
        else:
            body["max_tokens"] = 8000
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]["content"]
        if not msg.strip():
            raise RuntimeError("empty completion (reasoning model may have spent the whole "
                               "token budget on reasoning; raise max_completion_tokens)")
        return msg
    raise SystemExit(f"unknown provider: {provider}")


def iter_instances(limit: int | None):
    files = sorted(glob.glob(str(FT / "instances" / "*.json")))
    if limit:
        files = files[:limit]
    for f in files:
        yield Path(f), json.loads(Path(f).read_text())


def select_ids(per_commodity: int | None):
    """First N instance ids per commodity (balanced subset), or all."""
    by_base = {}
    for f in sorted(glob.glob(str(FT / "instances" / "*.json"))):
        stem = Path(f).stem
        by_base.setdefault(stem.split("__")[0], []).append(stem)
    out = []
    for base, ids in by_base.items():
        out += ids[:per_commodity] if per_commodity else ids
    return out


def dump_prompts(per_commodity: int | None) -> int:
    """Write each instance's authoring prompt to prompts/<id>.txt for a
    Claude-Code (or claude.ai) teacher to author into teacher_out/<id>.md."""
    PROMPTS.mkdir(parents=True, exist_ok=True)
    TEACHER_OUT.mkdir(parents=True, exist_ok=True)
    ids = select_ids(per_commodity)
    for iid in ids:
        inst = json.loads((FT / "instances" / f"{iid}.json").read_text())
        sysp, usr = build_full_prompt(inst)
        (PROMPTS / f"{iid}.txt").write_text(sysp + "\n\n===== TASK =====\n\n" + usr)
    print(f"[ok] wrote {len(ids)} prompts -> {PROMPTS}")
    print(f"     teacher authors each into {TEACHER_OUT}/<id>.md, then run --collect")
    return 0


def collect(val_frac: float) -> int:
    """Read teacher-authored DPRs from teacher_out/, assemble+verify each, and
    write curated train/val.jsonl. Only clean docs are kept."""
    rows, ok, bad = [], 0, 0
    for mdf in sorted(glob.glob(str(TEACHER_OUT / "*.md"))):
        iid = Path(mdf).stem
        ipath = FT / "instances" / f"{iid}.json"
        if not ipath.exists():
            continue
        inst = json.loads(ipath.read_text())
        doc = Path(mdf).read_text()
        _, rep = assemble(doc, inst)
        if rep["ok"]:
            ok += 1
            sysp, usr = build_full_prompt(inst)
            rows.append({"messages": [{"role": "system", "content": sysp},
                                      {"role": "user", "content": usr},
                                      {"role": "assistant", "content": doc}],
                         "meta": {"id": iid}})
        else:
            bad += 1
            print(f"  REJECT {iid}: unplaced={rep['unplaced_sections'][:3]} "
                  f"unknown={rep['unknown_placeholders']} num_issues={rep['number_issues'][:2]}")
    import random
    random.Random(13).shuffle(rows)          # deterministic; makes val commodity-diverse
    n_val = int(len(rows) * val_frac)
    (FT / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:n_val]) + "\n")
    (FT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows[n_val:]) + "\n")
    print(f"\n{ok} passed, {bad} rejected -> {len(rows)-n_val} train / {n_val} val in {FT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stand-in teacher, no API")
    ap.add_argument("--provider", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--show-prompt", help="print the prompt for one instance and exit")
    ap.add_argument("--dump-prompts", action="store_true",
                    help="write authoring prompts to prompts/ (teacher = Claude Code / claude.ai)")
    ap.add_argument("--collect", action="store_true",
                    help="read teacher_out/*.md, assemble+verify, write train/val.jsonl")
    ap.add_argument("--per-commodity", type=int, help="balanced subset size per commodity")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel API calls (--provider)")
    args = ap.parse_args()

    if args.show_prompt:
        inst = json.loads(Path(args.show_prompt).read_text())
        sysp, usr = build_full_prompt(inst)
        print("=== SYSTEM ===\n" + sysp + "\n\n=== USER ===\n" + usr)
        return 0
    if args.dump_prompts:
        return dump_prompts(args.per_commodity)
    if args.collect:
        return collect(args.val_frac)

    if not args.dry_run and not args.provider:
        print("give --dry-run, --dump-prompts, --collect, or --provider", file=sys.stderr)
        return 1

    if args.dry_run:
        # deterministic stub, no API — just prove the loop and drop a sample.
        ok = bad = 0; sample = None
        for path, inst in iter_instances(args.limit):
            final, rep = assemble(stub_teacher(inst), inst)
            ok += rep["ok"]; bad += (not rep["ok"])
            if rep["ok"] and sample is None:
                sample = final
        print(f"\n{ok} passed assemble+verify, {bad} rejected.")
        if sample:
            (FT / "sample_optionB.md").write_text(sample)
            print(f"[dry-run] loop proven; wrote a sample assembled DPR -> {FT/'sample_optionB.md'}")
        return 0

    # --provider: author with the frontier model IN PARALLEL, writing each clean DPR
    # to teacher_out/<id>.md. Skips already-authored ids (resumable across runs /
    # rate-limit interruptions). Assemble the dataset afterwards with --collect.
    import concurrent.futures as cf
    TEACHER_OUT.mkdir(parents=True, exist_ok=True)
    work = [(p, inst) for p, inst in iter_instances(args.limit)
            if not (TEACHER_OUT / f"{p.stem}.md").exists()]
    print(f"[author] {len(work)} to author (skipping existing) | model={args.model} "
          f"| concurrency={args.concurrency}")

    def worker(p, inst):
        sysp, usr = build_full_prompt(inst)
        doc = call_teacher(args.provider, args.model, sysp, usr)
        _, rep = assemble(doc, inst)
        return p.stem, doc, rep

    ok = bad = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(worker, p, inst): p.stem for p, inst in work}
        for fut in cf.as_completed(futs):
            iid = futs[fut]
            try:
                sid, doc, rep = fut.result()
            except Exception as e:
                bad += 1; print(f"  ERR {iid}: {type(e).__name__}: {str(e)[:90]}"); continue
            if rep["ok"]:
                (TEACHER_OUT / f"{sid}.md").write_text(doc); ok += 1
            else:
                bad += 1
                print(f"  REJECT {sid}: unplaced={len(rep['unplaced_sections'])} "
                      f"unknown={len(rep['unknown_placeholders'])} num_issues={len(rep['number_issues'])}")
    print(f"\n[author] {ok} clean -> teacher_out/, {bad} failed/rejected. "
          f"Next: ./venv/bin/python teacher.py --collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
