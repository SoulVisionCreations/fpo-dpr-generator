#!/usr/bin/env python3
"""
sft_train.py - LoRA supervised fine-tuning for the DPR narrative task.

Trains a student model (default gpt-oss-20b) to author the Option-B DPR document
(headings + §2 narrative + commentary + {{TABLE:id}} placeholders) from the
build_full_prompt() input. Loss is masked to the ASSISTANT turn only - the model
learns to GENERATE the document, never to predict the prompt. The engine still
owns every number: assembly + numeric verification happen deterministically at
inference (render.assemble), never here.

Data: data/finetune/train.jsonl / val.jsonl  (chat format: system/user/assistant),
produced by teacher.py --collect.

Setup (one-time):
    ./venv/bin/pip install peft datasets accelerate

Usage:
    # validate the data + masking without loading the 20B (fast, tokenizer only)
    ./venv/bin/python sft_train.py --check-data

    # train (run on the H200 via SLURM or directly)
    ./venv/bin/python sft_train.py --model openai/gpt-oss-20b --epochs 2
    ./venv/bin/python sft_train.py --model google/gemma-3-4b-it --epochs 3   # small student

Notes for gpt-oss-20b: loaded in bf16 (~40 GB, fits one H200); LoRA on the
attention projections keeps the base frozen. Set HF_HOME to the H200 disk as in
launch_pipeline.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dpr import config
FT = config.FINETUNE_DIR


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def tokenize_example(tok, messages: list[dict], max_len: int) -> dict | None:
    """Loss masked to the assistant turn only. The prompt (system+user, WITH the
    assistant generation cue) comes from the chat template and is fully masked;
    the assistant completion is tokenized directly and appended, so masking never
    depends on how the template renders the assistant turn. EOS is trained too."""
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    if not isinstance(prompt_ids, list):     # BatchEncoding / dict -> extract the id list
        prompt_ids = prompt_ids["input_ids"]
    comp_ids = tok(messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        comp_ids = comp_ids + [tok.eos_token_id]
    input_ids = (prompt_ids + comp_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + comp_ids)[:max_len]
    if not any(x != -100 for x in labels):
        return None
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def check_data(train: Path, val: Path, model: str, max_len: int) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    bad = False
    for name, p in [("train", train), ("val", val)]:
        if not p.exists():
            print(f"  {name}: MISSING ({p})")
            continue
        rows = load_rows(p)
        lens, kept = [], []
        for r in rows:
            ex = tokenize_example(tok, r["messages"], max_len)
            if ex:
                lens.append(len(ex["input_ids"]))
                kept.append(sum(1 for x in ex["labels"] if x != -100))
        valid = len(lens)
        extra = (f" | seq avg {sum(lens)//valid} max {max(lens)} | label(assistant) tokens avg {sum(kept)//valid}"
                 if valid else "")
        print(f"  {name}: {len(rows)} rows -> {valid} valid examples{extra}")
        if rows and valid == 0:
            bad = True
    if bad:
        print("\n[check] FAILED: a split tokenized to 0 valid examples (masking/template issue).")
        return 1
    print("\n[check] OK: assistant completion is the only loss target.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--train", type=Path, default=FT / "train.jsonl")
    ap.add_argument("--val", type=Path, default=FT / "val.jsonl")
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--check-data", action="store_true", help="validate data + masking, no training")
    args = ap.parse_args()

    if not args.train.exists() and not args.check_data:
        print(f"no {args.train} yet - run teacher.py --collect first", file=sys.stderr)
        return 1
    if args.check_data:
        return check_data(args.train, args.val, args.model, args.max_seq_len)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(path: Path):
        rows = load_rows(path)
        exs = [e for r in rows if (e := tokenize_example(tok, r["messages"], args.max_seq_len))]
        return Dataset.from_list(exs)

    train_ds = build(args.train)
    if len(train_ds) == 0:
        raise SystemExit("train dataset tokenized to 0 examples - run `sft_train.py --check-data` to debug masking")
    eval_ds = build(args.val) if args.val.exists() else None
    print(f"[data] train {len(train_ds)}" + (f" | val {len(eval_ds)}" if eval_ds else ""))

    from transformers import AutoConfig
    qc = getattr(AutoConfig.from_pretrained(args.model), "quantization_config", None)
    qmethod = (qc.get("quant_method") if isinstance(qc, dict)
               else getattr(qc, "quant_method", "")) if qc else ""
    is_mxfp4 = str(qmethod).lower() == "mxfp4"

    n_gpu = torch.cuda.device_count()
    free = {i: max(1, int(torch.cuda.mem_get_info(i)[0] / (1024 ** 3)) - 6) for i in range(n_gpu)}
    print(f"[load] {n_gpu} GPU(s); usable per-GPU (GiB): {free}")

    load_kwargs = dict(dtype=torch.bfloat16)
    if is_mxfp4:
        # MXFP4 dequant can't shard across GPUs (illegal memory access) -> single GPU.
        best = max(free, key=free.get)
        from transformers import Mxfp4Config
        load_kwargs["device_map"] = {"": best}
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print(f"[load] MXFP4 -> dequantize to bf16 on single GPU cuda:{best}")
    else:
        # Use only GPUs with real free room (skip ones a neighbour is hogging), so
        # the LM head + its ~18 GB CE spike can't land on a near-full GPU. Spread
        # weights thinly (small cap) so each keeps runtime headroom for the spike.
        # The loss spike materialises wherever the LM head sits - the FIRST device
        # for tied-embedding models (gemma), the LAST for untied ones (Qwen). So
        # rather than special-casing, only use GPUs that still have >= SPIKE free
        # AFTER their weight share, so the spike fits wherever it lands.
        SPIKE, cap = 32, 8
        roomy = {i: g for i, g in free.items() if g - cap >= SPIKE}
        if len(roomy) >= 2:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {i: f"{cap}GiB" for i in roomy}
            budget = cap * len(roomy)
            print(f"[load] weights across GPUs {sorted(roomy)} @ {cap} GiB each "
                  f"(budget {budget} GiB); every one keeps >= {SPIKE} GiB free for the loss spike")
        elif roomy:
            best = max(roomy, key=roomy.get)
            load_kwargs["device_map"] = {"": best}
            print(f"[load] single roomy GPU cuda:{best} ({roomy[best]} GiB free)")
        else:
            raise SystemExit(f"no GPU with >= {SPIKE} GiB free (node too contended); "
                             f"free per visible GPU (GiB): {free}")
    # NOTE: Liger's fused cross-entropy is numerically broken on this transformers
    # build (produces loss=0 / grad_norm=nan), so we use STANDARD CE (healthy) and
    # rely on multi-GPU weight-sharding above to give its ~18 GB logit spike room.
    # SDPA avoids the O(seq^2) eager-attention matrix; gpt-oss (eager/flash only)
    # falls back to eager.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", **load_kwargs)
        print("[load] attn_implementation=sdpa")
    except (ValueError, ImportError) as e:
        print(f"[load] sdpa unavailable ({type(e).__name__}); using default (eager) attention")
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False
    model.enable_input_require_grads()   # required for grad flow under checkpointing + frozen base

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",   # eval passes labels -> full-size CE -> OOM; evaluate after training instead
        remove_unused_columns=False,   # our dataset is pre-tokenized (input_ids/labels/mask)
        report_to=[],
    )
    # Under model-parallel sharding the loss lands on the last GPU while the
    # Trainer's num_items_in_batch counter sits on cuda:0 -> device mismatch in
    # Liger's `loss / num_items_in_batch`. Pass it as a plain int so the division
    # works on any device, keeping Liger's fused CE active (which avoids the OOM).
    class _MPTrainer(Trainer):
        # Standard (correct) loss; num_items_in_batch as a plain int so the
        # `loss / num_items_in_batch` division works across model-parallel devices.
        # (A logits_to_keep variant was tried to shrink the CE spike but produced a
        # misaligned loss ~= ln(vocab), so it was reverted.)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if hasattr(num_items_in_batch, "item"):
                num_items_in_batch = int(num_items_in_batch.item())
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    trainer = _MPTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))
    print(f"[ok] LoRA adapter -> {args.out/'adapter'}")
    print("Inference: load base + adapter, generate the DPR, then run it through "
          "render.assemble (inject exact tables + verify numbers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
