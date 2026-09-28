# FPO DPR generator — markdown-only inference service (vLLM + FastAPI).
#
# The official vLLM image ships a matched CUDA / torch / vLLM / flash-attn stack.
# Building that by hand on nvidia/cuda means owning the compatibility matrix.
#
# PIN THIS TAG. vLLM's Python API moves across minor releases, and an unpinned
# rebuild months from now will not start.
#
# v0.30.0 is the version gate0_check.py was verified against on the L40S box:
# Gemma4UnifiedForConditionalGeneration is in its registry, and the
# AsyncLLMEngine / TokensPrompt / LoRARequest calls in serve/engine.py all work.
# If you change this tag, re-run scripts/gate0_check.py --load --lora first.
FROM vllm/vllm-openai:v0.30.0

# The base image's entrypoint launches vLLM's own OpenAI server. We run our own
# FastAPI app instead, so clear it — forgetting this is the classic mistake.
ENTRYPOINT []

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # NOT optional: financial_engine._assert_balance is the only check that means
    # of finance equals project cost, and `python -O` strips it silently.
    PYTHONOPTIMIZE=0 \
    HF_HOME=/models/hf \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DPR_ADAPTER_DIR=/adapters/v4 \
    DPR_DATA_DIR=/app/data

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml ./
COPY dpr/ ./dpr/
COPY serve/ ./serve/
# Only templates — form_to_instance reads data/templates/<line>.json per request.
# The finetune corpus (25 MB) is training data and has no place in the image.
COPY data/templates/ ./data/templates/

# --no-deps: the image's torch must not be touched. `-e` is load-bearing —
# config.REPO_ROOT is derived from dpr/__file__, so a non-editable install would
# put dpr/ in site-packages and leave data/templates unreachable. DPR_DATA_DIR
# above is the belt-and-braces fix for the same hazard.
RUN pip install --no-deps -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --workers 1: N workers would mean N engines and instant OOM.
# --no-access-log: serve.middleware emits the structured access log.
# keep-alive above the 240s request deadline so the connection outlives a slow DPR.
CMD ["uvicorn", "serve.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "305", \
     "--no-access-log"]
