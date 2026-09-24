"""The shared core: retry policy, verification, and CLI/API equivalence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpr import config, pipeline
from dpr.backends.base import AdapterNotFoundError

BASE = Path(__file__).resolve().parents[1]
INSTANCE = json.loads((BASE / "data/finetune/instances/flour_mill__g004.json").read_text())


@pytest.fixture
def instance():
    return json.loads(json.dumps(INSTANCE))          # fresh copy per test


# ------------------------------------------------------------------- retries
def test_clean_first_attempt(good_generator, instance):
    r = pipeline.run({}, good_generator, instance=instance)
    assert r.attempt_count == 1 and r.clean
    assert good_generator.specs[0].greedy


def test_retry_after_unclean(flaky_generator, instance):
    r = pipeline.run({}, flaky_generator, instance=instance, seed=42)
    assert r.attempt_count == 2 and r.clean
    first, second = flaky_generator.specs
    assert first.greedy and first.top_p is None          # greedy omits top_p
    assert second.temperature == config.TEMPERATURE
    assert second.top_p == config.TOP_P
    assert second.seed == 43                             # seed + attempt index


def test_exhausted_retries_return_best_effort(broken_generator, instance):
    """Never raise — the CLI writes best-effort output, and so must the API."""
    r = pipeline.run({}, broken_generator, instance=instance, retries=2)
    assert r.attempt_count == 3
    assert not r.clean
    assert len(r.markdown) > 1000
    assert r.report["unplaced_sections"]                 # says *why* it is unclean


def test_retries_zero_means_one_attempt(broken_generator, instance):
    r = pipeline.run({}, broken_generator, instance=instance, retries=0)
    assert r.attempt_count == 1


def test_token_accounting(flaky_generator, instance):
    r = pipeline.run({}, flaky_generator, instance=instance)
    assert r.output_tokens == 2402 * 2                   # summed across attempts
    assert r.prompt_tokens == 1110                       # not summed


# ------------------------------------------------------------ sync/async parity
@pytest.mark.anyio
async def test_arun_matches_run(instance):
    from tests.conftest import GOOD_DOC, FakeGenerator

    sync = pipeline.run({}, FakeGenerator([GOOD_DOC]), instance=json.loads(json.dumps(instance)))
    a = await pipeline.arun({}, FakeGenerator([GOOD_DOC]), instance=json.loads(json.dumps(instance)))
    assert sync.markdown == a.markdown
    assert sync.report == a.report
    assert sync.attempt_count == a.attempt_count


@pytest.mark.anyio
async def test_arun_retry_policy_matches(instance):
    from tests.conftest import BROKEN_DOC, GOOD_DOC, FakeGenerator

    f = FakeGenerator([BROKEN_DOC, GOOD_DOC])
    r = await pipeline.arun({}, f, instance=instance, seed=7)
    assert r.attempt_count == 2
    assert f.specs[0].greedy
    assert (f.specs[1].temperature, f.specs[1].top_p, f.specs[1].seed) == (0.7, 0.9, 8)


# ------------------------------------------------------------------- prepare
def test_prepare_computes_and_balances(payload):
    inst = pipeline.prepare(payload)
    v = pipeline.viability(inst)
    assert inst["computed"]["project_cost"]["total_project_cost"] > 0
    assert v["balances"] is True


def test_viability_flags_are_data_not_prints(payload):
    v = pipeline.viability(pipeline.prepare(payload))
    assert set(v) == {"pat_year1", "average_dscr", "total_project_cost_lakh",
                      "balances", "financially_viable"}


# ------------------------------------------------------------------- backend
def test_missing_adapter_raises_not_exits(tmp_path):
    """SystemExit would be a BaseException and would kill a server worker."""
    from dpr.backends.hf import load_model
    with pytest.raises(AdapterNotFoundError):
        load_model("google/gemma-4-12B-it", tmp_path / "nope")


def test_adapter_error_is_catchable_as_exception(tmp_path):
    from dpr.backends.hf import load_model
    try:
        load_model("google/gemma-4-12B-it", tmp_path / "nope")
    except Exception as e:                                # the thing a server does
        assert isinstance(e, AdapterNotFoundError)
    else:
        pytest.fail("expected AdapterNotFoundError")


def test_adapter_check_precedes_weight_loading(tmp_path, monkeypatch):
    """It must fail before pulling 24 GB, not after."""
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "torch":
            pytest.fail("torch imported before the adapter check")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    from dpr.backends.hf import load_model
    with pytest.raises(AdapterNotFoundError):
        load_model("google/gemma-4-12B-it", tmp_path / "nope")


@pytest.fixture
def anyio_backend():
    return "asyncio"
