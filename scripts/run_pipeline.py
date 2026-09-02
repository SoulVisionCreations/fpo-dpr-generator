#!/usr/bin/env python3
"""
run_pipeline.py — end-to-end DPR generator: a flat website form payload in, a
finished DPR **PDF** out. One command, the whole production pipeline:

  flat JSON (form)                                                → DPR .pdf
     └─ form_to_instance.convert  ─ engine instance
        └─ financial_engine.compute  ─ all numbers
           └─ authoring.build_full_prompt → finetuned gemma (v4 LoRA) → DPR w/ {{TABLE}}
              └─ render/assemble.assemble  ─ inject exact engine tables + verify()   (retry if unclean)
                 └─ make_pdf (pandoc + tectonic)  ─ PDF

The engine owns every number; the model writes only prose; verify() guards every
figure. If a generation is structurally broken or leaks a number, it is re-sampled
(up to --retries) so a bad DPR is never written.

Usage:
  ./venv/bin/python run_pipeline.py --payload my_form.json
  ./venv/bin/python run_pipeline.py --payload my_form.json --out output/my_dpr.pdf
  ./venv/bin/python run_pipeline.py --payload my_form.json --md-only     # skip PDF (no pandoc/tectonic)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]              # repo root
sys.path.insert(0, str(BASE))                           

from dpr.form_to_instance import convert                # noqa: E402
from dpr.engine.financial_engine import compute         # noqa: E402
from dpr.authoring import build_full_prompt             # noqa: E402
from dpr.render.assemble import assemble                # noqa: E402
from dpr.make_pdf import (normalize_tables, rebuild_process_flow,   # noqa: E402
                          inject_toc_after_summary, PANDOC, TECTONIC, STYLE, NOTOC)

DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_ADAPTER = BASE / "models" / "google_gemma-4-12B-it_v4" / "adapter"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (s or "dpr")).strip("_") or "dpr"


def load_model(model_id: str, adapter: str | None):
    """Load base model (+ optional LoRA adapter) on the roomiest GPU, else CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        free = {i: int(torch.cuda.mem_get_info(i)[0] / 1024 ** 3) for i in range(torch.cuda.device_count())}
        best = max(free, key=free.get)
        load_kwargs = dict(dtype=torch.bfloat16, device_map={"": best})
        print(f"[load] cuda:{best} ({free[best]} GiB free)")
    else:
        load_kwargs = dict(dtype=torch.float32, device_map={"": "cpu"})
        print("[load] no GPU — running on CPU (slow)")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, KeyError, RuntimeError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    if adapter and Path(adapter).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"[load] adapter: {adapter}")
    elif adapter:
        print(f"[warn] adapter not found ({adapter}) — using BASE model")
    model.eval()
    return tok, model


def generate(tok, model, inst: dict, max_new_tokens: int, sample: bool) -> str:
    import torch
    sysp, usr = build_full_prompt(inst)
    chat = [{"role": "system", "content": sysp}, {"role": "user", "content": usr}]
    enc = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(model.device)
    attn = torch.ones_like(ids)          # single unpadded sequence -> attend to every token
    gen = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen.update(dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False))
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn, **gen)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def render_pdf(final_md: str, pdf_out: Path) -> bool:
    """Markdown DPR -> PDF via the project's pandoc + tectonic toolchain."""
    src = pdf_out.with_suffix(".md")
    src.write_text(inject_toc_after_summary(rebuild_process_flow(normalize_tables(final_md))), encoding="utf-8")
    env = dict(os.environ)
    env["HOME"] = str(BASE / ".tectonic-home")
    env["XDG_CACHE_HOME"] = str(BASE / ".tectonic-home" / ".cache")
    r = subprocess.run([str(PANDOC), str(src), "--defaults", str(STYLE), "--defaults", str(NOTOC),
                        "--pdf-engine", str(TECTONIC), "-o", str(pdf_out)], env=env, capture_output=True, text=True)
    if r.returncode != 0 or not pdf_out.exists():
        print("[pdf] FAILED:", (r.stderr.strip().splitlines() or ["?"])[-1][:200])
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Flat form payload -> DPR PDF (full pipeline).")
    ap.add_argument("--payload", required=True, type=Path, help="flat website form JSON")
    ap.add_argument("--out", type=Path, help="output PDF path (default: output/DPR_<product>_<ts>.pdf)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="LoRA adapter dir ('none' for base model)")
    ap.add_argument("--max-new-tokens", type=int, default=3500)
    ap.add_argument("--retries", type=int, default=2, help="re-sample if assemble/verify is unclean")
    ap.add_argument("--md-only", action="store_true", help="stop at markdown; skip PDF rendering")
    args = ap.parse_args()

    # 1) payload -> instance -> numbers
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    inst = convert(payload)
    compute(inst)
    c = inst["computed"]
    pc, mf, pr, d = c["project_cost"], c["means_of_finance"], c["profitability"], c["dscr"]
    print(f"[engine] {inst['meta']['commodity']} | TPC {pc['total_project_cost']}L | "
          f"PAT-y1 {pr['pat'][0]} | avg DSCR {d['average']} | balances "
          f"{abs(mf['total'] - pc['total_project_cost']) <= 0.05}")
    if pr["pat"][0] < 0 or d["average"] < 1.5:
        print("[engine] ⚠ project is not financially viable as configured (PAT<0 or DSCR<1.5) — "
              "DPR will still be generated, honestly reflecting the numbers.")

    # 2) author with the finetuned model, retry until assemble+verify is clean
    adapter = None if str(args.adapter).lower() == "none" else args.adapter
    tok, model = load_model(args.model, adapter)
    final, rep = None, None
    for attempt in range(args.retries + 1):
        doc = generate(tok, model, inst, args.max_new_tokens, sample=attempt > 0)
        final, rep = assemble(doc, inst)
        if rep["ok"]:
            print(f"[author] clean on attempt {attempt + 1} "
                  f"(placeholders placed, {len(rep['number_issues'])} number issues)")
            break
        print(f"[author] attempt {attempt + 1} unclean: unplaced={rep['unplaced_sections']} "
              f"unknown={rep['unknown_placeholders']} number_issues={len(rep['number_issues'])} — resampling")
    else:
        print("[author] ⚠ still unclean after retries — writing best effort (review flagged items).")

    # 3) write outputs
    out_pdf = args.out or (BASE / "output" / f"DPR_{_slug(inst['meta']['commodity'])}_"
                           f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_pdf.with_suffix(".md")
    md_path.write_text(final, encoding="utf-8")
    print(f"[md ] {md_path}")

    if args.md_only:
        print("[done] markdown only (--md-only).")
        return 0
    if render_pdf(final, out_pdf):
        print(f"[pdf] {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")
        print("[done] DPR ready.")
        return 0
    print("[done] markdown written but PDF rendering failed (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
