"""vLLM engine wrapper for the DPR narrative lane.

vLLM is imported lazily, inside `load()`. That is what lets serve.app — and the
whole CPU test suite — import cleanly on a machine with no GPU and no vLLM.

Two parity requirements are enforced here, both of which corrupt output silently
if missed:

1. STOP TOKENS. generation_config.json lists eos_token_id = [1, 106, 50]
   (<eos>, <turn|>, <|tool_response>) but the tokenizer alone reports only 1.
   The model ends its turn with <turn|> (106). HF's model.generate() reads
   generation_config and stops correctly; vLLM must be told explicitly or it
   generates to max_tokens on every request.

2. TOKENIZE ONCE. We hand vLLM token ids, not a string, so the prompt the model
   sees is exactly what the HF tokenizer produced — the same bytes the CLI feeds
   model.generate(). This tokenizer sets add_bos_token=False, so re-tokenizing
   the rendered string happens to be safe today (verified: one <bos> either way),
   but that is a property of the current tokenizer config, not a guarantee.
   Passing ids removes the question entirely.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dpr import config
from dpr.backends.base import AdapterNotFoundError, GenerationOutput, SamplingSpec

log = logging.getLogger("dpr.serve.engine")

MAX_MODEL_LEN = int(os.environ.get("DPR_MAX_MODEL_LEN", 8192))
GPU_MEM_UTIL = float(os.environ.get("DPR_GPU_MEMORY_UTILIZATION", 0.90))
MAX_NUM_SEQS = int(os.environ.get("DPR_MAX_NUM_SEQS", 8))
REQUEST_TIMEOUT_S = float(os.environ.get("DPR_REQUEST_TIMEOUT_S", 240))
# Text-only. Set DPR_DISABLE_MM_LIMITS=1 to fall back to vLLM defaults if a
# version rejects these keys (their names track the model's modalities).
MM_LIMITS = None if os.environ.get("DPR_DISABLE_MM_LIMITS") == "1" \
    else {"image": 0, "audio": 0, "video": 0}


@dataclass
class EngineState:
    """Lifecycle of the engine, surfaced by /health and /ready.

    Loading 24 GB takes minutes, so the app must serve health checks while it
    happens rather than blocking startup.
    """
    status: str = "loading"          # loading | warming | ready | failed
    error: str | None = None
    started_at: float = 0.0
    ready_at: float | None = None

    @property
    def elapsed_s(self) -> int:
        end = self.ready_at or time.time()
        return int(end - (self.started_at or end))

    @property
    def ready(self) -> bool:
        return self.status == "ready"


class VLLMGenerator:
    """Async backend satisfying dpr.backends.base.AsyncGenerator."""

    def __init__(self, engine, tokenizer, lora_request, model_id: str):
        self.engine = engine
        self.tok = tokenizer
        self.lora_request = lora_request
        self.model_id = model_id

    # ------------------------------------------------------------------ loading
    @classmethod
    async def load(cls, *, model_id: str | None = None, adapter: str | Path | None = None,
                   revision: str | None = None) -> "VLLMGenerator":
        model_id = model_id or config.MODEL_ID
        adapter = Path(adapter or config.DEFAULT_ADAPTER)
        revision = revision or config.MODEL_REVISION

        # Fail before pulling 24 GB of weights. In a service a silent fallback to
        # the base model is far worse than in a one-shot CLI run: it would serve
        # degraded output indefinitely with only a startup warning.
        if not adapter.is_dir():
            raise AdapterNotFoundError(adapter)

        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest

        # Prompt construction uses the adapter's tokenizer so the container needs
        # no separate copy of the base repo's tokenizer files. Verified
        # byte-identical to the base for tokenizer.json and chat_template.jinja.
        tok = AutoTokenizer.from_pretrained(str(adapter))

        args = AsyncEngineArgs(
            model=model_id,
            revision=revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=GPU_MEM_UTIL,
            # max_position_embeddings is 262144; left at the default vLLM would
            # reserve KV cache for a 256K context and OOM or waste the card.
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            enable_lora=True,
            max_lora_rank=16,               # matches adapter_config.json
            max_loras=1,
            max_cpu_loras=1,
            trust_remote_code=False,
            # This checkpoint carries vision AND audio towers. We only ever send
            # text, so zero the modality budgets: otherwise vLLM profiles the
            # encoders at startup ("profiled with 1 video items of the maximum
            # feature size") and reserves an encoder cache we never use.
            limit_mm_per_prompt=MM_LIMITS,
        )
        engine = AsyncLLMEngine.from_engine_args(args)
        lora = LoRARequest("dpr_v4", 1, str(adapter))
        log.info("vLLM engine constructed (model=%s adapter=%s max_model_len=%d)",
                 model_id, adapter, MAX_MODEL_LEN)
        return cls(engine, tok, lora, model_id)

    # ------------------------------------------------------------------- prompt
    def encode(self, system: str, user: str) -> list[int]:
        """Chat-template once, to token ids — never a string (see module docstring).

        apply_chat_template returns either a bare list or a BatchEncoding
        depending on transformers version and tokenizer; unwrap both, or we would
        hand vLLM the dict's KEYS ("input_ids", "attention_mask") as token ids.
        """
        chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if ids and isinstance(ids[0], (list, tuple)):      # batched shape [[...]]
            ids = ids[0]
        return [int(i) for i in ids]

    def _sampling(self, spec: SamplingSpec, logprobs: int | None = None):
        from vllm import SamplingParams
        kwargs = dict(
            max_tokens=spec.max_new_tokens,
            stop_token_ids=list(config.STOP_TOKEN_IDS),   # [1, 106, 50] — see docstring
            skip_special_tokens=True,                     # matches the CLI's decode
        )
        if spec.greedy:
            kwargs["temperature"] = 0.0                   # no top_p alongside greedy
        else:
            kwargs.update(temperature=spec.temperature, top_p=spec.top_p)
            if spec.seed is not None:
                kwargs["seed"] = spec.seed
        if logprobs:
            kwargs["logprobs"] = logprobs
        return SamplingParams(**kwargs)

    # --------------------------------------------------------------- generation
    async def agenerate(self, system: str, user: str, spec: SamplingSpec,
                        logprobs: int | None = None) -> GenerationOutput:
        from vllm import TokensPrompt

        ids = self.encode(system, user)
        request_id = f"dpr-{time.time_ns()}"
        params = self._sampling(spec, logprobs)

        final = None
        try:
            async for out in self.engine.generate(
                TokensPrompt(prompt_token_ids=ids), params, request_id,
                lora_request=self.lora_request,
            ):
                final = out
        except (asyncio.CancelledError, asyncio.TimeoutError):
            # A client that hung up must not leave a 3500-token decode burning GPU.
            await self.engine.abort(request_id)
            raise

        if final is None or not final.outputs:
            raise RuntimeError("vLLM returned no output")
        comp = final.outputs[0]
        first_lp = None
        if logprobs and comp.logprobs:
            # comp.logprobs[0] maps token_id -> Logprob for the first generated step
            first_lp = tuple(sorted((lp.logprob for lp in comp.logprobs[0].values()),
                                    reverse=True))
        return GenerationOutput(
            text=comp.text,
            prompt_tokens=len(final.prompt_token_ids or ids),
            output_tokens=len(comp.token_ids or []),
            finish_reason=comp.finish_reason or "",
            first_logprobs=first_lp,
        )

    async def warmup(self) -> None:
        """Pay CUDA-graph capture and LoRA-slot load here, not on the first request."""
        await self.agenerate("You are a test.", "Say ok.", SamplingSpec(max_new_tokens=4))

    async def shutdown(self) -> None:
        for name in ("shutdown", "shutdown_background_loop"):
            fn = getattr(self.engine, name, None)
            if fn:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
                return
