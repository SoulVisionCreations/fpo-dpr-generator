"""The /info discovery document, and /run as a third name for the generator.

The point of these tests is drift: /info exists so integrators do not have to
read the code, which only helps if it cannot fall out of step with the code.
"""
from __future__ import annotations

KEY = {"X-API-Key": "test-key"}


# ------------------------------------------------------------------------ /info
def test_info_needs_a_key(client):
    assert client.get("/info").status_code == 401


def test_info_answers_before_the_engine_is_ready(client, monkeypatch):
    """An integrator reading the contract should not wait 5 minutes for weights."""
    from serve import app as appmod
    monkeypatch.setattr(appmod.STATE, "status", "loading")
    r = client.get("/info", headers=KEY)
    assert r.status_code == 200
    assert r.json()["name"] == "fpo-dpr-generator"


def test_info_lists_every_business_line_the_converter_accepts(client):
    """The card is generated from BL_TO_TEMPLATE; assert it, so a 13th line added
    to the engine cannot ship with a card that still advertises twelve."""
    from dpr.form_to_instance import BL_TO_TEMPLATE
    body = client.get("/info", headers=KEY).json()
    assert {b["businessLine"] for b in body["business_lines"]} == set(BL_TO_TEMPLATE)


def test_info_reports_the_annual_basis_lines_correctly(client):
    """The error this document exists to prevent: sending a Storage / Warehouse
    capacity as a daily figure yields a balanced DPR with a negative DSCR."""
    body = client.get("/info", headers=KEY).json()
    basis = {b["businessLine"]: b["capacity_basis"] for b in body["business_lines"]}
    assert basis["Storage / Warehouse"] == "per year"
    assert basis["Custom Hiring Centre"] == "per year"
    assert basis["Rice Mill"] == "per day"


def test_info_capacity_defaults_match_the_engine(client):
    from dpr.form_to_instance import BL_TO_TEMPLATE, LINE_DEFAULTS
    body = client.get("/info", headers=KEY).json()
    for b in body["business_lines"]:
        d = LINE_DEFAULTS[BL_TO_TEMPLATE[b["businessLine"]]]
        assert b["capacity_default"] == d["cap"]
        assert b["capacity_unit"] == d["unit"]


def test_info_marks_business_line_as_the_only_required_field(client):
    body = client.get("/info", headers=KEY).json()
    assert body["input"]["required"] == ["businessLine"]
    required = [f["name"] for f in body["input"]["fields"] if f.get("required")]
    assert required == ["businessLine"]


def test_info_advertises_only_paths_that_exist(client):
    """Every path the card names must actually be routed."""
    from serve.app import app
    routed = {r.path for r in app.routes}
    for ep in client.get("/info", headers=KEY).json()["endpoints"].values():
        for path in ep["paths"]:
            assert path in routed, f"/info advertises {path}, which is not routed"


def test_info_output_schema_tracks_the_response_model(client):
    from serve.schemas import GenerateResponse
    body = client.get("/info", headers=KEY).json()
    assert body["output"]["json_schema"] == GenerateResponse.model_json_schema()


def test_info_schemes_match_the_engine(client):
    from dpr.form_to_instance import SCHEME_MAP
    assert client.get("/info", headers=KEY).json()["schemes"] == sorted(SCHEME_MAP)


# ------------------------------------------------------------------------- /run
def test_run_generates(client, payload):
    r = client.post("/run", json={"payload": payload}, headers=KEY)
    assert r.status_code == 200
    assert r.json()["markdown"].startswith("#")


def test_run_needs_a_key(client, payload):
    assert client.post("/run", json={"payload": payload}).status_code == 401


def test_run_is_the_same_handler_as_dpr(client, payload):
    """Three paths, one handler. If they ever diverge, this catches it."""
    bodies = []
    for url in ("/run", "/dpr", "/v1/dpr:generate"):
        r = client.post(url, json={"payload": payload}, headers=KEY)
        assert r.status_code == 200, url
        b = r.json()
        b.pop("request_id")                     # minted per request
        b["meta"].pop("generation_ms")          # wall-clock
        bodies.append(b)
    assert bodies[0] == bodies[1] == bodies[2]


def test_run_is_503_before_the_engine_is_ready(client, payload, monkeypatch):
    from serve import app as appmod
    monkeypatch.setattr(appmod.STATE, "status", "loading")
    r = client.post("/run", json={"payload": payload}, headers=KEY)
    assert r.status_code == 503
