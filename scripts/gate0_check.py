#!/usr/bin/env python3
"""Gate 0 — can this vLLM serve this model, with this LoRA?

Run this on the GPU box BEFORE building the image. If the architecture check
fails, no amount of wrapper code helps and the serving approach has to change.

  python scripts/gate0_check.py                 # checks only (no weights pulled)
  python scripts/gate0_check.py --load          # also builds the engine (~24 GB)
  python scripts/gate0_check.py --load --lora   # + proves the adapter is applied

Exit code 0 = gate passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

ARCH = "Gemma4UnifiedForConditionalGeneration"


def ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def bad(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def info(msg: str) -> None:
    print(f"        {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="actually build the engine (downloads ~24 GB)")
    ap.add_argument("--lora", action="store_true", help="with --load, prove the adapter changes output")
    args = ap.parse_args()

    failures = 0

    # ---------------------------------------------------------------- 1. vLLM
    print("\n[1] vLLM")
    try:
        import vllm
        ok(f"vllm {vllm.__version__}")
    except ImportError as e:
        bad(f"vllm not importable: {e}")
        return 1

    # -------------------------------------------------------- 2. architecture
    print(f"\n[2] architecture support — {ARCH}")
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        archs = set(ModelRegistry.get_supported_archs())
        if ARCH in archs:
            ok(f"{ARCH} is in the registry")
        else:
            bad(f"{ARCH} is NOT supported by vllm {vllm.__version__}")
            near = sorted(a for a in archs if "emma" in a)
            info(f"gemma architectures this build does support: {near or 'none'}")
            info("=> STOP. vLLM cannot serve this model; report before changing approach.")
            failures += 1
    except Exception as e:
        bad(f"could not read the registry: {type(e).__name__}: {e}")
        failures += 1

    # ------------------------------------------------------------- 3. adapter
    print("\n[3] adapter")
    from dpr import config
    adapter = Path(config.DEFAULT_ADAPTER)
    if adapter.is_dir():
        cfg = json.loads((adapter / "adapter_config.json").read_text())
        ok(f"{adapter}")
        info(f"r={cfg['r']} targets={cfg['target_modules']} base={cfg['base_model_name_or_path']}")
        if cfg["r"] > 16:
            bad(f"max_lora_rank in serve/engine.py is 16 but this adapter is r={cfg['r']}")
            failures += 1
    else:
        bad(f"adapter dir missing: {adapter}  (set DPR_ADAPTER_DIR)")
        failures += 1

    # --------------------------------------------------------- 4. stop tokens
    print("\n[4] stop tokens")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(adapter))
        names = [tok.convert_ids_to_tokens(i) for i in config.STOP_TOKEN_IDS]
        ok(f"config.STOP_TOKEN_IDS = {config.STOP_TOKEN_IDS} -> {names}")
        if tok.eos_token_id == config.STOP_TOKEN_IDS[0] and len(config.STOP_TOKEN_IDS) > 1:
            info("the tokenizer alone reports only <eos>; vLLM MUST be given all three,")
            info("or every request generates to max_tokens. serve/engine.py passes them.")
    except Exception as e:
        bad(f"tokenizer: {type(e).__name__}: {e}")
        failures += 1

    # -------------------------------------------------------------- 5. GPU
    print("\n[5] GPU")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            gib = total / 1024 ** 3
            ok(f"{torch.cuda.get_device_name(0)} — {gib:.1f} GiB total, {free / 1024**3:.1f} GiB free")
            budget = gib * float(os.environ.get("DPR_GPU_MEMORY_UTILIZATION", 0.90))
            kv = budget - 22.3 - 2.8
            info(f"at util 0.90: ~{budget:.1f} GiB budget - 22.3 weights - 2.8 overhead "
                 f"=> ~{kv:.1f} GiB KV cache")
            if kv < 4:
                bad("KV headroom looks too small — lower DPR_MAX_MODEL_LEN before raising util")
                failures += 1
        else:
            bad("no CUDA device visible")
            failures += 1
    except Exception as e:
        bad(f"torch: {type(e).__name__}: {e}")
        failures += 1

    if failures:
        print(f"\n=> GATE FAILED ({failures} issue(s)). Fix these before building the image.\n")
        return 1
    print("\n=> static checks passed."
          + ("" if args.load else "  Re-run with --load to build the engine.\n"))

    # ------------------------------------------------- 6. real engine + LoRA
    if args.load:
        print("\n[6] building the engine (this pulls ~24 GB on a cold cache)")
        import asyncio

        async def run() -> int:
            from dpr.backends.base import SamplingSpec
            from serve.engine import VLLMGenerator
            gen = await VLLMGenerator.load()
            ok("engine built")

            spec = SamplingSpec(max_new_tokens=64)
            out = await gen.agenerate("You are a helpful assistant.",
                                      "Name three Indian oilseeds.", spec)
            ok(f"generated {out.output_tokens} tokens, finish_reason={out.finish_reason!r}")
            if out.finish_reason == "length":
                bad("hit the token cap on a trivial prompt — stop tokens are NOT working")
                return 1
            info(f"sample: {out.text[:120]!r}")

            if args.lora:
                print("\n[7] proving the LoRA is actually applied")
                with_lora = out.text
                saved, gen.lora_request = gen.lora_request, None
                base = (await gen.agenerate("You are a helpful assistant.",
                                            "Name three Indian oilseeds.", spec)).text
                gen.lora_request = saved
                if with_lora.strip() == base.strip():
                    bad("identical output with and without the adapter — LoRA is NOT applied")
                    return 1
                ok("output differs with vs without the adapter — LoRA is live")
            await gen.shutdown()
            return 0

        return asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
