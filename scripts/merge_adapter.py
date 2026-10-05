#!/usr/bin/env python3
"""Merge the v4 LoRA adapter into the base model, producing standalone weights.

The result is a plain model directory that vLLM serves without `enable_lora`,
and that can be baked into a Docker image so the container needs no network and
no Hugging Face access at runtime.

Runs on CPU — this is weight arithmetic, not inference. It wants roughly 30 GB of
RAM and 50 GB of free disk (24 GB for the base, 24 GB for the merged copy).

  python scripts/merge_adapter.py                       # -> models/merged/
  python scripts/merge_adapter.py --out /data/merged    # custom destination
  python scripts/merge_adapter.py --verify              # + reload and sanity-check

Exit code 0 means the merged directory is complete and loadable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from dpr import config  # noqa: E402

# Tokenizer/template files vLLM needs beside the weights. They are copied from
# the adapter, which ships byte-identical copies of the base model's.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge the LoRA adapter into base weights.")
    ap.add_argument("--model", default=config.MODEL_ID, help="base model id or local path")
    ap.add_argument("--adapter", default=str(config.DEFAULT_ADAPTER))
    ap.add_argument("--out", type=Path, default=BASE / "models" / "merged")
    ap.add_argument("--verify", action="store_true", help="reload the merged model afterwards")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.is_dir():
        sys.exit(f"[error] adapter not found: {adapter}")
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        sys.exit(f"[error] {args.out} is not empty (use --force to overwrite)")

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    print(f"[plan] base    : {args.model}")
    print(f"[plan] adapter : {adapter}  (r={cfg['r']}, targets={cfg['target_modules']})")
    print(f"[plan] output  : {args.out}")

    expected_base = cfg.get("base_model_name_or_path")
    if expected_base and expected_base != args.model and not Path(args.model).exists():
        print(f"[warn] adapter was trained against {expected_base!r}, merging into "
              f"{args.model!r} — this is almost certainly wrong")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print("\n[1/4] loading base weights on CPU (several minutes, ~24 GB)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    except (ValueError, KeyError, RuntimeError):
        # This checkpoint is multimodal; CausalLM refuses it on some versions.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    print(f"      loaded in {time.time() - t0:.0f}s")

    print("[2/4] applying adapter")
    model = PeftModel.from_pretrained(model, str(adapter), device_map="cpu")

    print("[3/4] merging LoRA into the base weights")
    model = model.merge_and_unload()

    print(f"[4/4] writing to {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    # 5 GB shards: small enough that each sits well inside any registry layer
    # cap if the image's weights layer ever has to be split.
    model.save_pretrained(str(args.out), safe_serialization=True, max_shard_size="5GB")

    # The tokenizer must sit beside the weights: the image has no network, so
    # vLLM cannot fall back to fetching it from the Hub.
    tok = AutoTokenizer.from_pretrained(str(adapter))
    tok.save_pretrained(str(args.out))
    for name in TOKENIZER_FILES:
        src = adapter / name
        if src.is_file() and not (args.out / name).is_file():
            shutil.copy2(src, args.out / name)

    # generation_config carries eos_token_id = [1, 106, 50]. Losing it would mean
    # the model never stops on <turn|> and runs to the token cap every request.
    gen_cfg = args.out / "generation_config.json"
    if not gen_cfg.is_file():
        print("[warn] generation_config.json missing from the merged output —")
        print("       stop tokens must then be supplied explicitly by the server")

    print(f"\n[done] {dir_size(args.out) / 1024 ** 3:.2f} GB in {time.time() - t0:.0f}s")
    for f in sorted(args.out.iterdir()):
        if f.is_file():
            print(f"       {f.name:<36} {human(f.stat().st_size)}")

    if args.verify:
        print("\n[verify] reloading the merged model and checking the tokenizer")
        from transformers import AutoTokenizer as AT
        t = AT.from_pretrained(str(args.out))
        ids = t.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=True, add_generation_prompt=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        assert len(ids) > 3 and ids[0] == t.bos_token_id, "chat template did not apply"
        print(f"[verify] chat template OK ({len(ids)} tokens, leading BOS present)")
        if gen_cfg.is_file():
            eos = json.loads(gen_cfg.read_text()).get("eos_token_id")
            print(f"[verify] generation_config eos_token_id = {eos}")
            if eos != config.STOP_TOKEN_IDS:
                print(f"[verify] NOTE: differs from config.STOP_TOKEN_IDS "
                      f"{config.STOP_TOKEN_IDS}; the server passes its own list anyway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
