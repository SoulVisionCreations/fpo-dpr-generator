"""HTTP surface: auth, readiness, verification signalling, concurrency, errors.

All of it runs on CPU with a fake generator — no GPU, no vLLM.
"""
from __future__ import annotations

import asyncio

import pytest

KEY = {"X-API-Key": "test-key"}
URL = "/v1/dpr:generate"


# ------------------------------------------------------------------------ auth
def test_no_key_is_401(client, payload):
    assert client.post(URL, json={"payload": payload}).status_code == 401


def test_wrong_key_is_401(client, payload):
    r = client.post(URL, json={"payload": payload}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_bearer_token_accepted(client, payload):
    r = client.post(URL, json={"payload": payload},
                    headers={"Authorization": "Bearer test-key"})
    assert r.status_code == 200


def test_health_and_ready_need_no_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_auth_runs_before_a_slot_is_taken(client, payload):
    """An unauthenticated flood must not exhaust the concurrency budget."""
    from serve import middleware as mw
    before = mw.in_flight()
    for _ in range(10):
        client.post(URL, json={"payload": payload})
    assert mw.in_flight() == before


# -------------------------------------------------------------------- happy path
def test_generate_returns_markdown_and_verification(client, payload):
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 200
    b = r.json()
    assert b["markdown"].lstrip().startswith("#")
    assert b["verification"]["ok"] is True
    # the report must survive verbatim — all four keys
    assert set(b["verification"]) == {"ok", "unknown_placeholders",
                                      "unplaced_sections", "number_issues"}
    assert b["meta"]["attempts"] == 1
    assert b["meta"]["prompt_tokens"] == 1110
    assert b["meta"]["total_project_cost_lakh"] > 0


def test_engine_numbers_match_the_library(client, payload):
    from dpr import pipeline
    inst = pipeline.prepare(payload)
    v = pipeline.viability(inst)
    m = client.post(URL, json={"payload": payload}, headers=KEY).json()["meta"]
    assert m["total_project_cost_lakh"] == v["total_project_cost_lakh"]
    assert m["average_dscr"] == v["average_dscr"]
    assert m["balances"] is True


def test_request_id_is_echoed_and_minted(client, payload):
    r = client.post(URL, json={"payload": payload}, headers={**KEY, "X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["request_id"] == "abc123"
    r2 = client.post(URL, json={"payload": payload}, headers=KEY)
    assert len(r2.headers["X-Request-ID"]) == 32


def test_instance_is_omitted_unless_requested(client, payload):
    assert client.post(URL, json={"payload": payload}, headers=KEY).json()["instance"] is None
    r = client.post(URL, json={"payload": payload, "options": {"include_instance": True}},
                    headers=KEY)
    assert r.json()["instance"]["computed"]["project_cost"]["total_project_cost"] > 0


# --------------------------------------------------------------- verification
def test_unclean_is_200_with_a_header(client, payload, monkeypatch, broken_generator):
    """Best-effort output still ships, as the CLI does — signalled, not withheld."""
    from serve import app as appmod
    monkeypatch.setattr(appmod, "GENERATOR", broken_generator)
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 200
    assert r.headers["X-DPR-Verification"] == "unclean"
    b = r.json()
    assert b["verification"]["ok"] is False
    assert b["meta"]["clean"] is False
    assert len(b["markdown"]) > 1000


def test_strict_is_422_but_keeps_the_markdown(client, payload, monkeypatch, broken_generator):
    from serve import app as appmod
    monkeypatch.setattr(appmod, "GENERATOR", broken_generator)
    r = client.post(URL, json={"payload": payload, "options": {"strict": True}}, headers=KEY)
    assert r.status_code == 422
    assert len(r.json()["markdown"]) > 1000          # nothing is lost


def test_retry_then_clean(client, payload, monkeypatch, flaky_generator):
    from serve import app as appmod
    monkeypatch.setattr(appmod, "GENERATOR", flaky_generator)
    b = client.post(URL, json={"payload": payload}, headers=KEY).json()
    assert b["meta"]["attempts"] == 2 and b["verification"]["ok"] is True
    assert flaky_generator.specs[0].greedy
    assert flaky_generator.specs[1].temperature == 0.7
    assert flaky_generator.specs[1].top_p == 0.9


# ------------------------------------------------------------------- warnings
def test_unknown_business_line_is_flagged(client):
    """convert_website silently defaults; the API must not stay silent too."""
    bad = {"document_type": "dpr",
           "fields": {"businessLine": "Nonsense Co", "capacity": "500", "salePrice": "40"}}
    w = client.post(URL, json={"payload": bad}, headers=KEY).json()["warnings"]
    assert any("Nonsense Co" in x and "warehouse_cold_storage" in x for x in w)


def test_clean_payload_has_no_warnings(client, payload):
    assert client.post(URL, json={"payload": payload}, headers=KEY).json()["warnings"] == []


# ------------------------------------------------------------------ readiness
def test_generate_is_503_before_ready(client, payload, monkeypatch):
    from serve import app as appmod
    monkeypatch.setattr(appmod.STATE, "status", "loading")
    r = client.post(URL, json={"payload": payload}, headers=KEY)
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "60"


def test_health_stays_200_while_loading(client, monkeypatch):
    from serve import app as appmod
    monkeypatch.setattr(appmod.STATE, "status", "loading")
    assert client.get("/health").status_code == 200        # liveness: never restart mid-load
    assert client.get("/ready").status_code == 503         # readiness: keep traffic away


def test_ready_reports_failure(client, monkeypatch):
    from serve import app as appmod
    monkeypatch.setattr(appmod.STATE, "status", "failed")
    monkeypatch.setattr(appmod.STATE, "error", "AdapterNotFoundError: /adapters/v4")
    b = client.get("/ready").json()
    assert b["state"] == "failed" and "AdapterNotFound" in b["error"]


# --------------------------------------------------------------------- limits
def test_oversized_body_is_413(client, payload):
    r = client.post(URL, json={"payload": {"blob": "x" * 2_000_000}}, headers=KEY)
    assert r.status_code == 413


def test_unknown_envelope_field_is_422(client, payload):
    r = client.post(URL, json={"payload": payload, "nonsense": 1}, headers=KEY)
    assert r.status_code == 422


def test_arbitrary_form_fields_are_accepted(client):
    """payload is opaque on purpose — the API accepts whatever --payload accepts."""
    r = client.post(URL, json={"payload": {"fields": {"businessLine": "Rice Mill",
                                                      "whateverKey": "whateverValue"}}},
                    headers=KEY)
    assert r.status_code == 200


@pytest.mark.anyio
async def test_capacity_returns_429(monkeypatch, payload, good_generator):
    """A full queue must reject fast, not hold the connection."""
    import httpx
    from serve import app as appmod
    from serve import middleware as mw

    monkeypatch.setattr(appmod, "GENERATOR", good_generator)
    monkeypatch.setattr(appmod.STATE, "status", "ready")
    monkeypatch.setattr(mw, "_slots", asyncio.Semaphore(1))
    monkeypatch.setattr(mw, "QUEUE_WAIT_S", 0.05)

    original = good_generator.agenerate          # capture BEFORE patching, or slow() recurses

    async def slow(*a, **k):
        await asyncio.sleep(0.5)
        return await original(*a, **k)
    monkeypatch.setattr(good_generator, "agenerate", slow)

    transport = httpx.ASGITransport(app=appmod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        rs = await asyncio.gather(*[ac.post(URL, json={"payload": payload}, headers=KEY)
                                    for _ in range(3)])
    codes = sorted(r.status_code for r in rs)
    assert 429 in codes, codes
    assert 200 in codes, codes          # the admitted request must succeed, not time out
    assert 504 not in codes, codes
    assert any(r.headers.get("Retry-After") for r in rs if r.status_code == 429)


@pytest.fixture
def anyio_backend():
    return "asyncio"
