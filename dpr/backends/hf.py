"""HuggingFace transformers backend — the CLI's narrative lane.

Moved verbatim from scripts/run_pipeline.py, with one behavioural change: a
missing adapter raises AdapterNotFoundError instead of calling sys.exit, so the
same code is safe to call from a long-running process.
"""
from __future__ import annotations

from pathlib import Path

from dpr import config
from dpr.backends.base import AdapterNotFoundError, GenerationOutput, SamplingSpec


def load_model(model_id: str, adapter: str | Path | None, *, quiet: bool = False):
    """Load base model (+ optional LoRA adapter) on the roomiest GPU, else CPU."""
    # Check the adapter BEFORE pulling ~24 GB of base weights, so a missing
    # download fails in a second rather than after the whole model loads.
    if adapter and not Path(adapter).is_dir():
        raise AdapterNotFoundError(adapter)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    tok = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        free = {i: int(torch.cuda.mem_get_info(i)[0] / 1024 ** 3) for i in range(torch.cuda.device_count())}
        best = max(free, key=free.get)
        load_kwargs = dict(dtype=torch.bfloat16, device_map={"": best})
        say(f"[load] cuda:{best} ({free[best]} GiB free)")
    else:
        load_kwargs = dict(dtype=torch.float32, device_map={"": "cpu"})
        say("[load] no GPU — running on CPU (slow)")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, KeyError, RuntimeError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        say(f"[load] adapter: {adapter}")
    model.eval()
    return tok, model


class HFGenerator:
    """Wraps a loaded (tokenizer, model) pair as a pipeline Generator."""

    def __init__(self, tok, model):
        self.tok = tok
        self.model = model

    @classmethod
    def load(cls, model_id: str | None = None, adapter: str | Path | None = None,
             *, quiet: bool = False) -> "HFGenerator":
        return cls(*load_model(model_id or config.MODEL_ID, adapter, quiet=quiet))

    def generate(self, system: str, user: str, spec: SamplingSpec) -> GenerationOutput:
        import torch
        chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt")
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(self.model.device)
        attn = torch.ones_like(ids)          # single unpadded sequence -> attend to every token
        gen = dict(max_new_tokens=spec.max_new_tokens,
                   pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        # Greedy omits top_p entirely; sampling mirrors the CLI's 0.7 / 0.9.
        gen.update(dict(do_sample=False) if spec.greedy else
                   dict(do_sample=True, temperature=spec.temperature, top_p=spec.top_p))
        if spec.seed is not None:
            torch.manual_seed(spec.seed)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=attn, **gen)
        new = out[0][ids.shape[-1]:]
        return GenerationOutput(
            text=self.tok.decode(new, skip_special_tokens=True),
            prompt_tokens=int(ids.shape[-1]),
            output_tokens=int(new.shape[-1]),
        )
