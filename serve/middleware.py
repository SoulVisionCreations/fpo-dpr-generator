"""Cross-cutting HTTP concerns: auth, request logging, error mapping, concurrency.

Each concern uses the mechanism that fits it, and the choices are deliberate:

  auth          -> dependency        (middleware would gate /health and /ready,
                                      which load-balancer checks cannot authenticate)
  access log    -> HTTP middleware   (must wrap everything, including auth
                                      rejections; only layer seeing final status)
  error mapping -> exception handlers(one place, uniform envelope, no per-route try)
  concurrency   -> dependency        (must gate only /generate and release in finally)
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("dpr.serve")

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

MAX_BODY_BYTES = int(os.environ.get("DPR_MAX_BODY_BYTES", 1_048_576))       # 1 MiB
MAX_CONCURRENCY = int(os.environ.get("DPR_MAX_CONCURRENCY", 4))
QUEUE_WAIT_S = float(os.environ.get("DPR_QUEUE_WAIT_S", 5))
LOG_BODIES = os.environ.get("DPR_LOG_BODIES") == "1"

_slots = asyncio.Semaphore(MAX_CONCURRENCY)


# --------------------------------------------------------------------------- auth
def _valid_keys() -> list[str]:
    raw = os.environ.get("DPR_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def auth_enabled() -> bool:
    return os.environ.get("DPR_REQUIRE_AUTH", "1") != "0"


def check_startup_auth() -> None:
    """Fail closed: refuse to boot unauthenticated unless explicitly opted out."""
    if auth_enabled() and not _valid_keys():
        raise RuntimeError(
            "DPR_API_KEY is not set. Set it, or set DPR_REQUIRE_AUTH=0 to run without auth."
        )


# Declared as a security scheme (not just a header read) so it appears in
# OpenAPI and Swagger UI renders an "Authorize" button. auto_error=False keeps
# the Authorization: Bearer fallback working.
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="Service API key (DPR_API_KEY)")


async def require_api_key(request: Request,
                          key: str | None = Security(_api_key_scheme)) -> None:
    """Constant-time key check. Never logs or echoes the key."""
    if not auth_enabled():
        return
    presented = key or request.headers.get("x-api-key") or ""
    if not presented:
        bearer = request.headers.get("authorization") or ""
        if bearer.lower().startswith("bearer "):
            presented = bearer[7:].strip()
    # The NUMBER of valid keys is not secret, so per-key constant-time compare is
    # sufficient; `any()` short-circuiting leaks nothing about the key material.
    if not any(hmac.compare_digest(presented, k) for k in _valid_keys()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")


# -------------------------------------------------------------------- concurrency
async def acquire_slot():
    """Bounded admission. A full queue returns 429 rather than holding the client.

    With 30-90s requests an unbounded queue just converts into client-side
    timeouts on connections we are still holding open.
    """
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "server at capacity; retry shortly",
            headers={"Retry-After": "30"},
        )
    try:
        yield
    finally:
        _slots.release()


def in_flight() -> int:
    return MAX_CONCURRENCY - _slots._value       # diagnostics only


# ------------------------------------------------------------------------ logging
class AccessLogMiddleware(BaseHTTPMiddleware):
    """One structured JSON line per request, merging route-supplied fields.

    The handler knows domain facts (tokens, attempts, verification); only this
    layer knows final status and total latency. Neither can emit the line alone,
    so the route writes into request.state.log_extra and we merge here.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id_var.set(rid)
        request.state.request_id = rid
        request.state.log_extra = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            self._emit(request, 500, started)
            raise
        response.headers["X-Request-ID"] = rid
        self._emit(request, status_code, started)
        return response

    def _emit(self, request: Request, status_code: int, started: float) -> None:
        entry = {
            "request_id": getattr(request.state, "request_id", "-"),
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            **getattr(request.state, "log_extra", {}),
        }
        log.info(json.dumps(entry, default=str))


class BodyLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(
                {"error": "payload_too_large",
                 "detail": f"body exceeds {MAX_BODY_BYTES} bytes",
                 "request_id": request_id_var.get()},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        return await call_next(request)


def body_fingerprint(raw: bytes) -> dict:
    """Log the shape of a payload, never its contents — it is client business data."""
    out = {"body_bytes": len(raw), "body_sha256": hashlib.sha256(raw).hexdigest()[:16]}
    if LOG_BODIES:
        out["body"] = raw.decode("utf-8", "replace")[:8000]
    return out


# ---------------------------------------------------------------- error mapping
def _envelope(code: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": code, "detail": detail, "request_id": request_id_var.get()},
        status_code=status_code,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Map library exceptions to HTTP once, centrally.

    Note SystemExit deliberately has NO handler: it is a BaseException that would
    kill the worker, which is exactly why dpr.backends raises
    AdapterNotFoundError instead of calling sys.exit.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return _envelope("invalid_request", str(exc.errors()[:5]), 422)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        # A dict detail is already a structured envelope (e.g. the list of missing
        # financial fields) — pass it through rather than str()-ing it into prose
        # the caller would have to parse back out.
        if isinstance(exc.detail, dict):
            body = {**exc.detail, "request_id": request_id_var.get()}
            body.setdefault("error", f"http_{exc.status_code}")
            resp = JSONResponse(body, status_code=exc.status_code)
        else:
            resp = _envelope(f"http_{exc.status_code}", str(exc.detail), exc.status_code)
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    # --- caller's fault: the payload does not describe a computable project ---
    @app.exception_handler(ValueError)
    async def _value(request: Request, exc: ValueError):
        log.warning("payload rejected: %s: %s", type(exc).__name__, exc)
        return _envelope("invalid_payload", str(exc), 422)

    @app.exception_handler(NotImplementedError)
    async def _not_impl(request: Request, exc: NotImplementedError):
        return _envelope("unsupported_scheme", str(exc), 422)

    @app.exception_handler(ZeroDivisionError)
    async def _zero(request: Request, exc: ZeroDivisionError):
        log.warning("degenerate configuration", exc_info=True)
        return _envelope("degenerate_configuration",
                         "the supplied parameters do not produce a computable project", 422)

    @app.exception_handler(KeyError)
    async def _key(request: Request, exc: KeyError):
        # Ambiguous: a missing payload field or a template bug. Return only the
        # key name; the traceback goes to the log for us, not to the caller.
        log.error("KeyError during generation", exc_info=True)
        return _envelope("missing_field", f"missing required field: {exc}", 422)

    # --- our fault: deploy or engine invariant ---
    @app.exception_handler(AssertionError)
    async def _assertion(request: Request, exc: AssertionError):
        log.error("engine invariant failed", exc_info=True)
        return _envelope("engine_invariant_failed",
                         "means of finance did not balance against project cost", 500)

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(request: Request, exc: FileNotFoundError):
        log.error("server misconfiguration: %s", exc, exc_info=True)
        return _envelope("server_misconfigured", "a required data file is missing", 500)

    @app.exception_handler(json.JSONDecodeError)
    async def _bad_json(request: Request, exc: json.JSONDecodeError):
        log.error("corrupt data file: %s", exc, exc_info=True)
        return _envelope("server_misconfigured", "a required data file is unreadable", 500)

    @app.exception_handler(asyncio.TimeoutError)
    async def _timeout(request: Request, exc: asyncio.TimeoutError):
        return _envelope("generation_timeout", "generation exceeded the deadline", 504)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return _envelope("internal_error", "internal error", 500)


def install(app: FastAPI) -> None:
    """Wire middleware (outermost first) and handlers."""
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(AccessLogMiddleware)     # added last == outermost
    install_exception_handlers(app)


__all__ = ["install", "require_api_key", "acquire_slot", "check_startup_auth",
           "body_fingerprint", "request_id_var", "in_flight", "Depends"]
