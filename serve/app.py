"""FastAPI service: flat form payload -> verified DPR markdown.

Markdown only — no pandoc/tectonic anywhere in this path.

The engine loads in a background task rather than in lifespan: uvicorn accepts no
connections until lifespan startup returns, so blocking it for the minutes it
takes to load 24 GB would make health checks fail before the app exists.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

# compute() ends in `assert diff <= 0.05` — the only guarantee that means of
# finance equals project cost. Under python -O that check vanishes silently, so
# refuse to serve rather than emit unbalanced DPRs.
if not __debug__:
    raise RuntimeError(
        "Refusing to start under PYTHONOPTIMIZE: the engine's balance assertion "
        "(financial_engine._assert_balance) would be stripped."
    )

from dpr import config, pipeline                                   # noqa: E402
from dpr.backends.base import AdapterNotFoundError                 # noqa: E402
from serve import middleware as mw                                 # noqa: E402
from serve.engine import REQUEST_TIMEOUT_S, EngineState            # noqa: E402
from serve.logging_config import configure as configure_logging    # noqa: E402
from serve.schemas import (GenerateRequest, GenerateResponse,      # noqa: E402
                           Meta, Verification)

log = logging.getLogger("dpr.serve")

STATE = EngineState()
GENERATOR = None


async def _load_engine() -> None:
    """Background engine load. Failures are recorded, never raised into startup."""
    global GENERATOR
    STATE.started_at = time.time()
    try:
        from serve.engine import VLLMGenerator          # lazy: keeps vLLM off the import path
        GENERATOR = await VLLMGenerator.load()
        STATE.status = "warming"
        await GENERATOR.warmup()
        STATE.status = "ready"
        STATE.ready_at = time.time()
        log.info("engine ready in %ss", STATE.elapsed_s)
    except Exception as e:                               # noqa: BLE001 - surfaced via /ready
        STATE.status = "failed"
        STATE.error = f"{type(e).__name__}: {e}"
        log.exception("engine failed to load")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()                                  # or every access line is discarded
    mw.check_startup_auth()                              # fail closed before serving
    task = asyncio.create_task(_load_engine())
    yield
    task.cancel()
    # getattr, not a bare call: a generator that failed partway through load — or
    # a test double — may not implement shutdown.
    closer = getattr(GENERATOR, "shutdown", None)
    if closer:
        await closer()


# Swagger/ReDoc describe the whole API surface to anyone who can reach the port.
# Handy while testing, unnecessary in production — DPR_ENABLE_DOCS=0 removes
# /docs, /redoc and /openapi.json entirely (404, not merely hidden).
ENABLE_DOCS = os.environ.get("DPR_ENABLE_DOCS", "1") != "0"

app = FastAPI(
    title="FPO DPR Generator",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)
mw.install(app)


async def require_ready() -> None:
    from fastapi import HTTPException
    if not STATE.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"engine not ready (state={STATE.status})",
            headers={"Retry-After": "60"},
        )


# ------------------------------------------------------------------ operational
@app.get("/health")
async def health() -> dict:
    """Liveness: 200 while the process lives, whatever the engine is doing.

    Restart only on process death — never because a 24 GB load is still running.
    """
    return {"status": "ok", "engine": STATE.status}


@app.get("/ready")
async def ready() -> Response:
    """Readiness: the load-balancer check. 200 only once warmup has completed."""
    body = {"state": STATE.status, "elapsed_s": STATE.elapsed_s, "error": STATE.error}
    code = status.HTTP_200_OK if STATE.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(body, status_code=code)


@app.get("/v1/info", dependencies=[Depends(mw.require_api_key)])
async def info() -> dict:
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = None
    from serve.engine import GPU_MEM_UTIL, MAX_MODEL_LEN
    return {
        "model": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "adapter": str(config.DEFAULT_ADAPTER),
        "vllm": vllm_version,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "max_new_tokens_default": config.MAX_NEW_TOKENS,
        "retries_default": config.RETRIES,
        "stop_token_ids": config.STOP_TOKEN_IDS,
        "max_concurrency": mw.MAX_CONCURRENCY,
        "in_flight": mw.in_flight(),
        "engine_state": STATE.status,
        "git_sha": os.environ.get("GIT_SHA"),
    }


# -------------------------------------------------------------------- generation
def _warnings(payload: dict, inst: dict) -> list[str]:
    """Flag silent defaulting.

    convert_website never raises: an unknown businessLine quietly becomes
    warehouse_cold_storage. Over HTTP that yields a confident, plausible, wrong
    DPR with no signal, so say so explicitly.
    """
    out: list[str] = []
    # Every figure the converter substituted for a blank form field. These are
    # deterministic line/template defaults, never model output — but they appear
    # in the DPR as fact, so the caller has to be able to see them.
    for d in inst.get("meta", {}).get("defaults_applied", []):
        unit = f" {d['unit']}" if d.get("unit") else ""
        out.append(f"{d['field']} not supplied — used {d['value']}{unit} ({d.get('source','default')})")
    # The engine already decides whether a scheme was modelled; read its note
    # rather than re-implementing the lookup and risking the two disagreeing.
    note = inst.get("computed", {}).get("means_of_finance", {}).get("fallback_note")
    if note:
        out.append(str(note))
    return out


@app.post("/v1/dpr:generate", response_model=GenerateResponse,
          dependencies=[Depends(require_ready),
                        Depends(mw.require_api_key),     # auth before a slot is consumed
                        Depends(mw.acquire_slot)])
async def generate(req: GenerateRequest, request: Request, response: Response) -> GenerateResponse:
    from fastapi import HTTPException

    rid = getattr(request.state, "request_id", "-")
    opts = req.options

    inst = pipeline.prepare(req.payload)
    v = pipeline.viability(inst)
    warnings = _warnings(req.payload, inst)
    applied = inst.get("meta", {}).get("defaults_applied", [])

    # Bank/NABARD submissions require every figure to originate from the applicant.
    # Reject before spending ~70s of GPU on a document that would be rejected anyway.
    if opts.require_complete_inputs and applied:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"error": "incomplete_inputs",
             "detail": "financial fields missing; no defaults applied because "
                       "require_complete_inputs was set",
             "missing": [d["field"] for d in applied],
             "defaults_that_would_have_been_used": applied})

    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            pipeline.arun(req.payload, GENERATOR, retries=opts.retries,
                          max_new_tokens=opts.max_new_tokens, seed=opts.seed,
                          instance=inst),
            timeout=REQUEST_TIMEOUT_S,
        )
    except AdapterNotFoundError as e:
        log.error("adapter missing at request time: %s", e.path)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model adapter unavailable")
    gen_ms = int((time.perf_counter() - t0) * 1000)

    rep = result.report
    meta = Meta(
        **pipeline.meta(inst),
        attempts=result.attempt_count,
        clean=result.clean,
        total_project_cost_lakh=v["total_project_cost_lakh"],
        average_dscr=v["average_dscr"],
        pat_year1=v["pat_year1"],
        balances=v["balances"],
        financially_viable=v["financially_viable"],
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        generation_ms=gen_ms,
        model=config.MODEL_ID,
        model_revision=config.MODEL_REVISION,
        adapter=config.DEFAULT_ADAPTER.name,
    )

    # Merged into the single access-log line by AccessLogMiddleware.
    request.state.log_extra = {
        "commodity": meta.commodity,
        "attempts": meta.attempts,
        "verify_ok": rep.get("ok"),
        "number_issues": len(rep.get("number_issues", [])),
        "unplaced": len(rep.get("unplaced_sections", [])),
        "prompt_tokens": meta.prompt_tokens,
        "output_tokens": meta.output_tokens,
        "generation_ms": gen_ms,
        "warnings": len(warnings),
        "defaults_applied": len(applied),
    }

    body = GenerateResponse(
        request_id=rid,
        markdown=result.markdown,
        verification=Verification(**{k: rep.get(k) for k in
                                     ("ok", "unknown_placeholders", "unplaced_sections", "number_issues")}),
        meta=meta,
        warnings=warnings,
        defaults_applied=applied,
        instance=inst if opts.include_instance else None,
    )

    if not result.clean:
        log.warning(json.dumps({"request_id": rid, "event": "unclean_verification",
                                "number_issues": rep.get("number_issues", [])[:5],
                                "unplaced_sections": rep.get("unplaced_sections")}))
        # strict callers get 422 — but the markdown still travels, so nothing is lost.
        if opts.strict:
            return JSONResponse(body.model_dump(), status_code=422,
                                headers={"X-DPR-Verification": "unclean"})
        response.headers["X-DPR-Verification"] = "unclean"
    return body
