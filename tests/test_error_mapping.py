"""Library exceptions must become the right HTTP status.

There is no schema validation in the pipeline — schema/dpr_input.schema.json is
documentation and is never loaded — so these handlers are the only thing
separating a caller's bad payload from a 500.
"""
from __future__ import annotations

import asyncio
import json

import pytest

KEY = {"X-API-Key": "test-key"}
URL = "/v1/dpr:generate"


def _raise(exc):
    def _boom(_payload):
        raise exc
    return _boom


@pytest.mark.parametrize("exc,expected,code", [
    (ValueError("capacity unit mismatch"), 422, "invalid_payload"),
    (NotImplementedError("Scheme 'smam' is a stub"), 422, "unsupported_scheme"),
    (ZeroDivisionError("division by zero"), 422, "degenerate_configuration"),
    (KeyError("working_days_per_annum"), 422, "missing_field"),
    (AssertionError("means of finance 12.0 != project cost 15.0"), 500, "engine_invariant_failed"),
    (FileNotFoundError("data/templates/rice_mill.json"), 500, "server_misconfigured"),
    (json.JSONDecodeError("bad", "{}", 0), 500, "server_misconfigured"),
])
def test_exception_maps_to_status(client, payload, monkeypatch, exc, expected, code):
    from dpr import pipeline
    from serve import app as appmod
    monkeypatch.setattr(appmod.pipeline, "prepare", _raise(exc))

    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == expected, r.text
    assert r.json()["error"] == code


def test_error_envelope_carries_request_id(client, payload, monkeypatch):
    from serve import app as appmod
    monkeypatch.setattr(appmod.pipeline, "prepare", _raise(ValueError("nope")))
    r = client.post(URL, json={"payload": payload},
                    headers={**KEY, "X-Request-ID": "trace-me"})
    assert r.json()["request_id"] == "trace-me"


def test_internal_detail_is_not_leaked(client, payload, monkeypatch):
    """A 500 must not hand the caller our internals."""
    from serve import app as appmod
    secret = "/opt/secret/path/weights.bin"
    monkeypatch.setattr(appmod.pipeline, "prepare", _raise(FileNotFoundError(secret)))
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 500
    assert secret not in r.text


def test_timeout_becomes_504(client, payload, monkeypatch):
    from serve import app as appmod

    async def _slow(*a, **k):
        raise asyncio.TimeoutError()
    monkeypatch.setattr(appmod.pipeline, "arun", _slow)
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 504
    assert r.json()["error"] == "generation_timeout"


def test_missing_adapter_at_request_time_is_503(client, payload, monkeypatch):
    """Not a 500: the adapter mount vanishing is an availability problem."""
    from dpr.backends.base import AdapterNotFoundError
    from serve import app as appmod

    async def _gone(*a, **k):
        raise AdapterNotFoundError("/adapters/v4")
    monkeypatch.setattr(appmod.pipeline, "arun", _gone)
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 503


def test_unhandled_exception_is_500_generic(payload, monkeypatch, good_generator):
    """Starlette's catch-all handler returns the 500 AND re-raises so the server
    logs it. TestClient surfaces that re-raise by default, so this test uses
    raise_server_exceptions=False to observe what a real client would receive."""
    from fastapi.testclient import TestClient
    from serve import app as appmod

    async def _no_load():
        return None
    monkeypatch.setattr(appmod, "_load_engine", _no_load)
    monkeypatch.setattr(appmod, "GENERATOR", good_generator)
    monkeypatch.setattr(appmod.STATE, "status", "ready")
    monkeypatch.setattr(appmod.pipeline, "prepare", _raise(RuntimeError("kaboom")))

    with TestClient(appmod.app, raise_server_exceptions=False) as c:
        r = c.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 500
    assert "kaboom" not in r.text, "internal detail leaked to the caller"
    assert r.json()["error"] == "internal_error"


def test_startup_refuses_without_api_key(monkeypatch):
    """Fail closed: never boot unauthenticated by accident."""
    from serve import middleware as mw
    monkeypatch.delenv("DPR_API_KEY", raising=False)
    monkeypatch.setenv("DPR_REQUIRE_AUTH", "1")
    with pytest.raises(RuntimeError, match="DPR_API_KEY"):
        mw.check_startup_auth()


def test_startup_allows_explicit_opt_out(monkeypatch):
    from serve import middleware as mw
    monkeypatch.delenv("DPR_API_KEY", raising=False)
    monkeypatch.setenv("DPR_REQUIRE_AUTH", "0")
    mw.check_startup_auth()          # must not raise
