"""Guarantees the partner (bank / NABARD) audit depends on.

Two distinct claims, tested separately:

  1. The LLM never contributes a financial figure. Tables come from the engine;
     the model's prose is re-read and any unmatched figure is flagged.
  2. Any figure the FPO did not supply is reported. Deterministic line defaults
     are legitimate, but they must never be silent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
KEY = {"X-API-Key": "test-key"}
URL = "/v1/dpr:generate"

SPARSE = {"document_type": "dpr",
          "fields": {"businessLine": "Rice Mill", "landStatus": "Owned",
                     "schemeChoice": "Yes", "schemeName": "PMFME"}}


# ------------------------------------------------- 1. no LLM-authored figures
def test_every_table_value_comes_from_the_engine(payload):
    """Tables are substituted wholesale; the model only places a marker."""
    from dpr import pipeline
    from dpr.render.assemble import section_catalog

    inst = pipeline.prepare(payload)
    cat = section_catalog(inst)
    from dpr.render.assemble import assemble
    doc = "# T\n\n## 1. Project Summary\n{{TABLE:1}}\n\n## 3. Basis\n{{TABLE:3}}\n"
    final, rep = assemble(doc, inst)
    # the engine's exact body is present, byte for byte
    assert cat["1"]["body"] in final
    assert "{{TABLE:1}}" not in final


def test_invented_figure_in_prose_is_flagged():
    """A number the model makes up must not pass verification."""
    from dpr import pipeline
    from dpr.render.assemble import assemble

    inst = pipeline.prepare(SPARSE)
    doc = ("# T\n\n## 2. Product\n\nThe unit will earn Rs. 987.65 lakh in year one.\n")
    _, rep = assemble(doc, inst)
    nums = [i["value"] for i in rep["number_issues"]]
    assert 987.65 in nums, "an invented figure escaped verification"
    assert rep["ok"] is False


def test_engine_figures_in_prose_are_accepted():
    """The guard must not fire on figures that ARE the engine's."""
    from dpr import pipeline
    from dpr.render.assemble import assemble

    inst = pipeline.prepare(SPARSE)
    tpc = inst["computed"]["project_cost"]["total_project_cost"]
    doc = f"# T\n\n## 2. Product\n\nTotal project cost is Rs. {tpc} lakh.\n"
    _, rep = assemble(doc, inst)
    assert rep["number_issues"] == [], rep["number_issues"]


def test_strict_blocks_a_document_with_an_unverified_figure(client, monkeypatch, payload):
    from serve import app as appmod
    from tests.conftest import FakeGenerator

    bad = "# T\n\n## 2. Product\n\nRevenue will reach Rs. 4321.99 lakh.\n"
    monkeypatch.setattr(appmod, "GENERATOR", FakeGenerator([bad]))
    r = client.post(URL, json={"payload": payload, "options": {"strict": True}}, headers=KEY)
    assert r.status_code == 422
    assert r.json()["verification"]["ok"] is False


# ------------------------------------------- 2. no silent financial defaults
def test_sparse_payload_reports_every_substitution():
    from dpr import pipeline
    applied = pipeline.prepare(SPARSE)["meta"]["defaults_applied"]
    fields = {d["field"] for d in applied}
    for expected in ("capacity", "workingDays", "salePrice", "rmcost",
                     "machinery", "electricityCost", "loan.tenure_months"):
        assert expected in fields, f"{expected} was substituted silently"


def test_complete_payload_substitutes_nothing():
    """The reference fixture supplies everything — the list must be empty."""
    from dpr import pipeline
    full = json.loads((BASE / "data/examples/mango_juice.json").read_text())
    assert pipeline.prepare(full)["meta"]["defaults_applied"] == []


def test_every_default_is_reported_in_warnings(client):
    from dpr import pipeline
    applied = pipeline.prepare(SPARSE)["meta"]["defaults_applied"]
    body = client.post(URL, json={"payload": SPARSE}, headers=KEY).json()
    assert len(body["defaults_applied"]) == len(applied)
    # every substituted field is named in the human-readable warnings too
    joined = " ".join(body["warnings"])
    for d in applied:
        assert d["field"] in joined


def test_defaults_applied_is_structured(client):
    d = client.post(URL, json={"payload": SPARSE}, headers=KEY).json()["defaults_applied"][0]
    assert set(d) == {"field", "value", "unit", "source"}


def test_require_complete_inputs_rejects_before_generating(client, monkeypatch):
    """Must 422 without burning GPU on a document that would be rejected anyway."""
    from serve import app as appmod

    called = False

    async def _should_not_run(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("generation ran despite incomplete inputs")

    monkeypatch.setattr(appmod.pipeline, "arun", _should_not_run)
    r = client.post(URL, json={"payload": SPARSE,
                               "options": {"require_complete_inputs": True}}, headers=KEY)
    assert r.status_code == 422
    assert called is False


def test_require_complete_inputs_passes_a_full_payload(client, payload):
    r = client.post(URL, json={"payload": payload,
                               "options": {"require_complete_inputs": True}}, headers=KEY)
    assert r.status_code == 200
    assert r.json()["defaults_applied"] == []


def test_defaults_never_come_from_the_model(client, payload, monkeypatch, broken_generator):
    """Even when the model misbehaves, defaults_applied reflects only the converter."""
    from serve import app as appmod
    monkeypatch.setattr(appmod, "GENERATOR", broken_generator)
    body = client.post(URL, json={"payload": SPARSE}, headers=KEY).json()
    sources = {d["source"] for d in body["defaults_applied"]}
    assert all("template" in s or "default" in s or "assumption" in s or "supplied" in s
               for s in sources), sources


def test_incomplete_inputs_error_is_machine_readable(client):
    """The caller must be able to parse which fields are missing, not scrape prose."""
    r = client.post(URL, json={"payload": SPARSE,
                               "options": {"require_complete_inputs": True}}, headers=KEY)
    b = r.json()
    assert b["error"] == "incomplete_inputs"
    assert isinstance(b["missing"], list) and "capacity" in b["missing"]
    assert b["defaults_that_would_have_been_used"][0]["field"] == "capacity"
    assert b["request_id"]


@pytest.mark.parametrize("scheme_name", [
    "RKVY", "State Horticulture Mission", "Rashtriya Krishi Vikas Yojana",
    "NABARD-AIF 2.0", "Others",
])
def test_unmodelled_scheme_note_names_it_as_the_fpo_typed_it(scheme_name):
    """The note is printed in the DPR that goes to the bank, so it must show the
    scheme's real name — not the slugified key we use for lookup."""
    from dpr import pipeline
    p = {"document_type": "dpr",
         "fields": {"businessLine": "Rice Mill", "schemeChoice": "Yes",
                    "schemeName": scheme_name}}
    mf = pipeline.prepare(p)["computed"]["means_of_finance"]
    assert f"'{scheme_name}'" in mf["fallback_note"]
    assert mf["subsidy"] == 0.0, "an unmodelled scheme must not assume a subsidy"
    assert "self-funded" in mf["scheme"].lower()


def test_unmodelled_scheme_note_reaches_the_document():
    """Not just the API response — §5.7 of the markdown must carry it."""
    from dpr import pipeline
    from dpr.render.sections import render
    p = {"document_type": "dpr",
         "fields": {"businessLine": "Rice Mill", "schemeChoice": "Yes",
                    "schemeName": "State Horticulture Mission"}}
    md = render(pipeline.prepare(p), narrative_md="x")
    body = md if isinstance(md, str) else "\n".join(s["body"] for s in md.values())
    assert "State Horticulture Mission" in body
    assert "not modelled in this tool" in body


def test_modelled_scheme_has_no_fallback_note():
    from dpr import pipeline
    p = {"document_type": "dpr",
         "fields": {"businessLine": "Rice Mill", "schemeChoice": "Yes", "schemeName": "PMFME"}}
    mf = pipeline.prepare(p)["computed"]["means_of_finance"]
    assert not mf.get("fallback_note")
    assert mf["subsidy"] > 0


# ------------------------------------------- merged-weights (bundled image) mode
def test_merged_mode_disables_lora(monkeypatch):
    """A bundled image bakes the adapter into the weights, so vLLM must not be
    told to load a LoRA on top — that would be a no-op at best."""
    import importlib
    monkeypatch.setenv("DPR_MERGED_MODEL", "1")
    import serve.engine as e
    importlib.reload(e)
    try:
        assert e.MERGED is True
        kwargs = {"enable_lora": not e.MERGED,
                  **({} if e.MERGED else {"max_lora_rank": 16})}
        assert kwargs == {"enable_lora": False}
    finally:
        monkeypatch.delenv("DPR_MERGED_MODEL", raising=False)
        importlib.reload(e)


def test_adapter_mode_is_the_default(monkeypatch):
    import importlib
    monkeypatch.delenv("DPR_MERGED_MODEL", raising=False)
    import serve.engine as e
    importlib.reload(e)
    assert e.MERGED is False


@pytest.mark.anyio
async def test_merged_mode_rejects_a_missing_weights_dir(monkeypatch):
    """Fail at startup with a clear message, not deep inside vLLM."""
    import importlib
    from dpr.backends.base import AdapterNotFoundError
    monkeypatch.setenv("DPR_MERGED_MODEL", "1")
    monkeypatch.setenv("DPR_MODEL_ID", "/models/does-not-exist")
    import serve.engine as e
    importlib.reload(e)
    try:
        with pytest.raises(AdapterNotFoundError, match="merged model directory"):
            await e.VLLMGenerator.load()
    finally:
        monkeypatch.delenv("DPR_MERGED_MODEL", raising=False)
        monkeypatch.delenv("DPR_MODEL_ID", raising=False)
        importlib.reload(e)


@pytest.fixture
def anyio_backend():
    return "asyncio"
