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

This is a thin CLI over `dpr.pipeline`, which the HTTP service shares — the model
lane lives in `dpr.backends.hf`.

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
sys.path.insert(0, str(BASE))                           # allow `import dpr` without pip install

from dpr import config, pipeline                        # noqa: E402
from dpr.backends.base import AdapterNotFoundError      # noqa: E402
from dpr.backends.hf import HFGenerator                 # noqa: E402
from dpr.make_pdf import (normalize_tables, rebuild_process_flow,   # noqa: E402
                          inject_toc_after_summary, PANDOC, TECTONIC, STYLE, NOTOC)

DEFAULT_MODEL = config.MODEL_ID
DEFAULT_ADAPTER = config.DEFAULT_ADAPTER


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (s or "dpr")).strip("_") or "dpr"


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
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    ap.add_argument("--retries", type=int, default=config.RETRIES, help="re-sample if assemble/verify is unclean")
    ap.add_argument("--md-only", action="store_true", help="stop at markdown; skip PDF rendering")
    args = ap.parse_args()

    # 1) payload -> instance -> numbers
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    inst = pipeline.prepare(payload)
    v = pipeline.viability(inst)
    print(f"[engine] {inst['meta']['commodity']} | TPC {v['total_project_cost_lakh']}L | "
          f"PAT-y1 {v['pat_year1']} | avg DSCR {v['average_dscr']} | balances "
          f"{v['balances']}")
    if not v["financially_viable"]:
        print("[engine] ⚠ project is not financially viable as configured (PAT<0 or DSCR<1.5) — "
              "DPR will still be generated, honestly reflecting the numbers.")

    # 2) author with the finetuned model, retry until assemble+verify is clean
    adapter = None if str(args.adapter).lower() == "none" else args.adapter
    try:
        generator = HFGenerator.load(args.model, adapter)
    except AdapterNotFoundError as e:
        sys.exit(f"[error] adapter not found: {e.path}\n"
                 f"        Download it (README -> 'Get the model') and place it there, pass\n"
                 f"        --adapter <dir> to point elsewhere, or --adapter none to run the\n"
                 f"        BASE model deliberately (noticeably lower quality).")

    result = pipeline.run(payload, generator, retries=args.retries,
                          max_new_tokens=args.max_new_tokens, instance=inst)
    for a in result.attempts:
        rep = a.report
        if rep["ok"]:
            print(f"[author] clean on attempt {a.index + 1} "
                  f"(placeholders placed, {len(rep['number_issues'])} number issues)")
        else:
            print(f"[author] attempt {a.index + 1} unclean: unplaced={rep['unplaced_sections']} "
                  f"unknown={rep['unknown_placeholders']} number_issues={len(rep['number_issues'])} — resampling")
    if not result.clean:
        print("[author] ⚠ still unclean after retries — writing best effort (review flagged items).")

    # 3) write outputs
    final = result.markdown
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
