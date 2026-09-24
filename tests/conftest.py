"""Shared fixtures. Everything here runs without a GPU or vLLM installed."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

os.environ.setdefault("DPR_API_KEY", "test-key")

from dpr.backends.base import GenerationOutput, SamplingSpec  # noqa: E402

GOOD_DOC = json.loads((BASE / "data/finetune/val.jsonl").read_text()
                      .splitlines()[0])["messages"][-1]["content"]
BROKEN_DOC = GOOD_DOC.replace("{{TABLE:5.7}}", "")      # a dropped placeholder -> unclean


@pytest.fixture
def payload() -> dict:
    return json.loads((BASE / "data/examples/mango_juice.json").read_text())


class FakeGenerator:
    """Records the SamplingSpec of every attempt so tests can assert retry policy."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.specs: list[SamplingSpec] = []

    def _next(self) -> str:
        return self.texts[min(len(self.specs) - 1, len(self.texts) - 1)]

    def generate(self, system: str, user: str, spec: SamplingSpec) -> GenerationOutput:
        self.specs.append(spec)
        return GenerationOutput(text=self._next(), prompt_tokens=1110, output_tokens=2402)

    async def agenerate(self, system: str, user: str, spec: SamplingSpec) -> GenerationOutput:
        return self.generate(system, user, spec)

    async def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def good_generator() -> FakeGenerator:
    return FakeGenerator([GOOD_DOC])


@pytest.fixture
def flaky_generator() -> FakeGenerator:
    return FakeGenerator([BROKEN_DOC, GOOD_DOC])


@pytest.fixture
def broken_generator() -> FakeGenerator:
    return FakeGenerator([BROKEN_DOC])


@pytest.fixture
def client(monkeypatch, good_generator):
    """TestClient with the engine faked and marked ready.

    _load_engine is stubbed out first: otherwise the lifespan's background task
    would try to import vLLM, fail, and race the STATE patch back to "failed".
    """
    from fastapi.testclient import TestClient
    from serve import app as appmod

    async def _no_load():
        return None

    monkeypatch.setattr(appmod, "_load_engine", _no_load)
    monkeypatch.setattr(appmod, "GENERATOR", good_generator)
    monkeypatch.setattr(appmod.STATE, "status", "ready")
    with TestClient(appmod.app) as c:
        c.fake = good_generator
        yield c
