#!/usr/bin/env python3
"""
eval_ft.py - evaluate a fine-tuned DPR model on held-out instances.

For each val instance: build_full_prompt -> generate the placeholder DPR ->
render.assemble (inject the exact engine tables + verify every number the model
typed) -> score the section-2 narrative. Reports the two things that matter:

  structure : did it place every {{TABLE:id}} exactly once (assemble.ok)?
  numbers   : did verify() find any figure that isn't an engine value?
  quality   : score_dpr dimensions on the §2 prose.

Compare fine-tuned vs base by running twice (with and without --adapter).

Usage:
  ./venv/bin/python eval_ft.py --model Qwen/Qwen2.5-7B-Instruct \
      --adapter models/Qwen_Qwen2.5-7B-Instruct/adapter --limit 3
  ./venv/bin/python eval_ft.py --model Qwen/Qwen2.5-7B-Instruct --limit 3   # base, no adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dpr import config
FT = config.FINETUNE_DIR

from dpr.authoring import build_full_prompt
from dpr.render.assemble import assemble
from training import score_dpr as S


def score_section2(doc: str, inst: dict) -> dict:
    """score_dpr dimensions, computed directly (we already hold the instance)."""
    s2 = S.extract_section2(doc)
    dims, detail = {}, {}
    dims["subsection_coverage"], detail["coverage"] = S.score_coverage(s2)
    dims["no_leaked_figures"], detail["leaks"] = S.score_leaks(s2)
    dims["conciseness"], detail["conciseness"] = S.score_conciseness(s2)
    dims["low_repetition"], detail["repetition"] = S.score_repetition(s2)
    dims["process_grounding"], detail["grounding"] = S.score_grounding(s2, inst)
    dims["business_model_fidelity"], detail["business_model"] = S.score_business_model(s2, inst)
    total = sum(dims[k] * S.WEIGHTS[k] for k in dims) / sum(S.WEIGHTS[k] for k in dims) * 100
    return {"total": round(total, 1), "dims": {k: round(v, 3) for k, v in dims.items()},
            "detail": detail}


def val_ids(limit: int | None) -> list[str]:
    p = FT / "val.jsonl"
    ids = [json.loads(l)["meta"]["id"] for l in p.read_text().splitlines() if l.strip()]
    return ids[:limit] if limit else ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter dir; omit to evaluate the base model")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=3500)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "ft_eval")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    n_gpu = torch.cuda.device_count()
    free = {i: max(1, int(torch.cuda.mem_get_info(i)[0] / (1024 ** 3)) - 6) for i in range(n_gpu)}
    print(f"[load] {n_gpu} GPU(s); usable (GiB): {free}")
    # Inference (no training loss spike): put the whole model on the single roomiest
    # GPU when it fits there. Sharding across GPUs forces a cross-GPU hop per token
    # and makes generation much slower - only worth it if the model can't fit on one.
    best = max(free, key=free.get)
    load_kwargs = dict(dtype=torch.bfloat16, device_map={"": best})
    print(f"[load] single GPU cuda:{best} ({free[best]} GiB free) - fast generation, no sharding")
    # Multimodal models (gemma-3n, gemma-3-4b, ...) don't load under
    # AutoModelForCausalLM; fall back to the image-text-to-text class, then to the
    # generic AutoModel. We only ever feed text, so any of them generates fine.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except (ValueError, KeyError, RuntimeError) as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}); trying multimodal loader")
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(args.model, **load_kwargs)
    # namespace outputs by model AND base/finetuned so parallel configs don't clobber
    mslug = args.model.replace("/", "_").replace(":", "_")
    tag = f"{mslug}__base"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = f"{mslug}__finetuned"
        print(f"[load] adapter: {args.adapter}")
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid in val_ids(args.limit):
        inst = json.loads((FT / "instances" / f"{iid}.json").read_text())
        sysp, usr = build_full_prompt(inst)
        chat = [{"role": "system", "content": sysp}, {"role": "user", "content": usr}]
        enc = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
        # may come back as a BatchEncoding rather than a plain tensor
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        doc = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        final, rep = assemble(doc, inst)
        sc = score_section2(doc, inst)
        (args.out / f"{iid}__{tag}.md").write_text(final)
        rows.append({"id": iid, "assemble_ok": rep["ok"],
                     "unplaced": len(rep["unplaced_sections"]), "unknown": len(rep["unknown_placeholders"]),
                     "number_issues": len(rep["number_issues"]), "s2_score": sc["total"]})
        r = rows[-1]
        print(f"  {iid:28} assemble_ok={r['assemble_ok']!s:5} unplaced={r['unplaced']:2} "
              f"unknown={r['unknown']:2} num_issues={r['number_issues']:2} §2={r['s2_score']:.1f}")

    n = len(rows) or 1
    print(f"\n[{tag}] {len(rows)} instances | assemble ok: {sum(r['assemble_ok'] for r in rows)}/{len(rows)}"
          f" | zero number issues: {sum(r['number_issues'] == 0 for r in rows)}/{len(rows)}"
          f" | mean §2 score: {sum(r['s2_score'] for r in rows)/n:.1f}")
    (args.out / f"summary__{tag}.json").write_text(json.dumps(rows, indent=2))
    print(f"[ok] DPRs + summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())