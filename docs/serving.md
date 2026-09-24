# Serving the DPR generator

A single FastAPI process with the vLLM engine in-process. Synchronous: the request
blocks until the markdown is ready (~30–90 s). **Markdown only** — no pandoc, no
tectonic, no PDF anywhere in this path.

```
POST /v1/dpr:generate  →  convert → compute → vLLM → assemble + verify  →  markdown
```

---

## Endpoints

| Route | Auth | Purpose |
|---|---|---|
| `POST /v1/dpr:generate` | yes | form payload → verified DPR markdown |
| `GET /health` | no | liveness — **always 200** while the process lives |
| `GET /ready` | no | readiness — 200 only once the engine has warmed |
| `GET /v1/info` | yes | model, adapter, vLLM version, limits, in-flight count |

`/health` and `/ready` are deliberately separate. Loading 24 GB takes minutes; a
single endpoint would make the load balancer kill the container mid-load. Point
liveness at `/health` and the target group at `/ready`, with a **startup grace
period of at least 20 minutes** for the first (uncached) boot.

### Request

```bash
curl -X POST http://host:8000/v1/dpr:generate \
  -H 'X-API-Key: $DPR_API_KEY' -H 'content-type: application/json' \
  -d '{"payload": {"document_type":"dpr","fields":{...}}, "options": {"retries":2}}'
```

`payload` is the same flat website form the CLI takes via `--payload`, and it is
deliberately **opaque** — the API accepts exactly what `--payload` accepts, so no
Pydantic model constrains the form fields. Unknown keys in the *envelope* are
rejected (422).

| Option | Default | Effect |
|---|---|---|
| `retries` | 2 | re-samples while assemble/verify is unclean |
| `max_new_tokens` | 3500 | matches the CLI, so output is comparable |
| `seed` | none | makes a flagged generation reproducible from the log |
| `strict` | false | unclean verification → 422 (markdown still returned) |
| `include_instance` | false | also return the computed engine instance |

### Response

```jsonc
{
  "request_id": "…",
  "markdown": "# DETAILED PROJECT REPORT — …",
  "verification": { "ok": true, "unknown_placeholders": [], "unplaced_sections": [], "number_issues": [] },
  "meta": { "commodity": "Mango Juice", "attempts": 1, "clean": true,
            "total_project_cost_lakh": 68.54, "average_dscr": 8.65, "financially_viable": true,
            "prompt_tokens": 1110, "output_tokens": 2402, "generation_ms": 41200 },
  "warnings": []
}
```

**`verification` is the engine's report verbatim.** The CLI prints it; hiding it
over HTTP would be strictly worse. An unclean generation still returns **200**
with `X-DPR-Verification: unclean` — the CLI writes best-effort output after
exhausting retries and the API preserves that contract. Use `strict: true` to get
a 422 instead; the markdown still travels in the body either way.

**`warnings` matters.** `convert_website` never raises: an unrecognised
`businessLine` silently becomes `warehouse_cold_storage`, and bad numbers fall
back to line defaults. Over HTTP that would hand a caller a confident, plausible,
wrong DPR with no signal, so any silent defaulting is reported here.

### Status codes

| Code | Meaning |
|---|---|
| 200 | markdown produced (check `verification.ok`) |
| 401 | missing or invalid API key |
| 413 | body over `DPR_MAX_BODY_BYTES` |
| 422 | payload not computable, or `strict` with unclean verification |
| 429 | at capacity — honour `Retry-After` |
| 503 | engine still loading, failed, or adapter unavailable |
| 504 | generation exceeded `DPR_REQUEST_TIMEOUT_S` |

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `DPR_API_KEY` | — | **required**; comma-separated for rotation. Startup fails without it |
| `DPR_REQUIRE_AUTH` | `1` | `0` disables auth — local use only |
| `DPR_MAX_CONCURRENCY` | 4 | in-flight generations; excess waits `DPR_QUEUE_WAIT_S` then 429 |
| `DPR_REQUEST_TIMEOUT_S` | 240 | deadline; the vLLM request is aborted on expiry |
| `DPR_MAX_MODEL_LEN` | 8192 | **do not raise casually** — see below |
| `DPR_GPU_MEMORY_UTILIZATION` | 0.90 | never above 0.93 on a headless card |
| `DPR_ADAPTER_DIR` | `/adapters/v4` | must exist at startup or the service refuses to boot |
| `DPR_MODEL_REVISION` | none | pin for reproducible output |
| `DPR_LOG_BODIES` | `0` | `1` logs full payloads — client business data, off by default |

---

## Deployment (L40S 48 GB, g6e.2xlarge)

```bash
docker build -t dpr-api:$(git rev-parse --short HEAD) .

docker run -d --name dpr-api --gpus all \
  --ipc=host --shm-size=8g --ulimit memlock=-1 \
  -p 8000:8000 --restart unless-stopped \
  -e DPR_API_KEY=... \
  -v /mnt/dpr/hf:/models/hf \
  -v /mnt/dpr/adapter:/adapters/v4:ro \
  dpr-api:$(git rev-parse --short HEAD)
```

**`--ipc=host` is required, not optional.** vLLM's V1 engine uses shared memory
between the API process and its EngineCore subprocess; Docker's default 64 MB
`/dev/shm` produces obscure crashes.

**Weights are mounted, never baked.** The base model is 23.92 GB — baking it
would mean re-pushing that layer through ECR on every code change. It downloads
once to `/mnt/dpr/hf` and is reused across restarts. The adapter is mounted
read-only from wherever you pulled it out of S3.

**Prefer g6e.2xlarge over g6e.xlarge**: weights stream through host RAM during
load, and 32 GB is tight. EBS ≥ 200 GB gp3.

### Memory

| | GiB |
|---|---|
| L40S usable | ~45.6 |
| budget at `gpu_memory_utilization=0.90` | ~41.0 |
| bf16 weights | −22.3 |
| CUDA graphs, activations, LoRA | −2.8 |
| **KV cache** | **≈ 15–16** |

`max_position_embeddings` is **262144**. Left at the default, vLLM reserves KV
cache for a 256 K context and either OOMs or wastes the card — hence
`max_model_len=8192`, which comfortably covers ~1.1 k prompt + 3.5 k output. Read
vLLM's `GPU KV cache size: N tokens` line at startup and calibrate. If it is
tight, lower `max_model_len` to 6144 **before** raising `gpu_memory_utilization`.

> **ALB idle timeout defaults to 60 s**, well below the 30–90 s response time.
> Raise it to 300 s, and tell clients to set `read_timeout ≥ 300`.

---

## Things that will silently corrupt output

**Stop tokens.** `generation_config.json` lists `eos_token_id = [1, 106, 50]` —
`<eos>`, `<turn|>`, `<|tool_response>` — but the tokenizer alone reports only `1`.
The model ends its turn with **`<turn|>` (106)**. HF's `model.generate()` reads
`generation_config` and stops correctly; vLLM must be told explicitly via
`stop_token_ids`, or it generates to `max_tokens` on **every** request and appends
garbage after the DPR. This lives in `config.STOP_TOKEN_IDS` and is asserted by
`tests/test_prompt_parity.py`.

**`PYTHONOPTIMIZE`.** `financial_engine._assert_balance` is the only guarantee
that means of finance equals project cost, and `python -O` strips assertions.
`serve/app.py` refuses to import under `-O`; keep `PYTHONOPTIMIZE=0` in the image.

**LoRA silently not applied.** vLLM can load and serve the base model while
ignoring the adapter. Before trusting a deployment, generate the same prompt with
and without the `LoRARequest` and confirm the outputs differ.

**`--workers 1`.** N uvicorn workers means N engines and instant OOM.

---

## Verifying a deployment

```bash
# 1. architecture support — run this BEFORE anything else
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
  print('Gemma4UnifiedForConditionalGeneration' in ModelRegistry.get_supported_archs())"

# 2. readiness
curl -fsS localhost:8000/ready

# 3. generate, and confirm it stopped on <turn|> rather than running to 3500
curl -s -X POST localhost:8000/v1/dpr:generate -H "X-API-Key: $DPR_API_KEY" \
  -H 'content-type: application/json' \
  -d @<(jq '{payload: .}' data/examples/mango_juice.json) | jq '.meta'

# 4. compare against the CLI on the same payload
python scripts/run_pipeline.py --payload data/examples/mango_juice.json --md-only
```

If `meta.output_tokens` is pinned at `max_new_tokens` on every request, the stop
tokens are not being honoured. If §2 prose is markedly worse than the CLI's, the
LoRA is not applied.

The CPU test suite (`pytest tests/`) runs without a GPU or vLLM installed and
covers the pipeline, prompt parity, the HTTP surface and error mapping.
