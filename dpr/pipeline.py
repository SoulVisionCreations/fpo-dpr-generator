"""Backend-agnostic DPR pipeline core, shared by the CLI and the HTTP service.

    payload -> prepare() -> instance (every number computed by the engine)
            -> run()     -> markdown (model prose + exact engine tables, verified)

The model authors prose and {{TABLE:id}} placeholders; assemble() injects the
engine's tables and verify() rejects any figure the model typed that the engine
did not produce. An unclean generation is re-sampled up to `retries` times.

No printing, no file I/O, no argparse — callers decide how to report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dpr import config
from dpr.authoring import build_full_prompt
from dpr.backends.base import AsyncGenerator, GenerationOutput, Generator, SamplingSpec
from dpr.engine.financial_engine import compute
from dpr.form_to_instance import convert
from dpr.render.assemble import assemble


@dataclass(frozen=True)
class Attempt:
    """What one generation asked for and what came back."""
    index: int
    spec: SamplingSpec
    report: dict
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class PipelineResult:
    instance: dict
    markdown: str
    report: dict
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when assemble+verify accepted the generation outright."""
        return bool(self.report.get("ok"))

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def prompt_tokens(self) -> int:
        return self.attempts[-1].prompt_tokens if self.attempts else 0

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)


def prepare(payload: dict) -> dict:
    """Form payload -> engine instance with every figure computed.

    `compute` mutates in place and returns the same object; we return it so
    callers need not rely on the mutation.
    """
    return compute(convert(payload))


def viability(instance: dict) -> dict:
    """The CLI's PAT/DSCR warning, as data rather than a printed line.

    A non-viable project still produces a DPR — honestly reflecting the numbers —
    so this is advisory, never a hard failure.
    """
    c = instance["computed"]
    pat_y1 = c["profitability"]["pat"][0]
    avg_dscr = c["dscr"]["average"]
    pc, mf = c["project_cost"], c["means_of_finance"]
    return {
        "pat_year1": pat_y1,
        "average_dscr": avg_dscr,
        "total_project_cost_lakh": pc["total_project_cost"],
        "balances": abs(mf["total"] - pc["total_project_cost"]) <= 0.05,
        "financially_viable": not (pat_y1 < 0 or avg_dscr < 1.5),
    }


def sampling_for_attempt(index: int, max_new_tokens: int, seed: int | None = None) -> SamplingSpec:
    """Attempt 0 greedy, retries sampled — the CLI's semantics, in one place.

    Greedy deliberately leaves top_p unset: passing it alongside temperature=0 is
    meaningless and some backends warn about the combination.
    """
    if index == 0:
        return SamplingSpec(max_new_tokens=max_new_tokens)
    return SamplingSpec(
        max_new_tokens=max_new_tokens,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        seed=None if seed is None else seed + index,
    )


def should_retry(report: dict) -> bool:
    return not report.get("ok")


def _specs(retries: int, max_new_tokens: int, seed: int | None):
    for i in range(retries + 1):
        yield i, sampling_for_attempt(i, max_new_tokens, seed)


def _finish(instance: dict, best: tuple[str, dict] | None, attempts: list[Attempt]) -> PipelineResult:
    # `best` is None only if the loop never ran (retries < 0), which argparse and
    # the API schema both prevent; guard anyway rather than raising IndexError.
    markdown, report = best if best else ("", {"ok": False, "unknown_placeholders": [],
                                               "unplaced_sections": [], "number_issues": []})
    return PipelineResult(instance=instance, markdown=markdown, report=report, attempts=attempts)


def run(payload: dict, generator: Generator, *, retries: int | None = None,
        max_new_tokens: int | None = None, seed: int | None = None,
        instance: dict | None = None) -> PipelineResult:
    """Synchronous: payload -> verified DPR markdown. Used by the CLI."""
    retries = config.RETRIES if retries is None else retries
    max_new_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    inst = instance if instance is not None else prepare(payload)
    system, user = build_full_prompt(inst)

    attempts: list[Attempt] = []
    best: tuple[str, dict] | None = None
    for i, spec in _specs(retries, max_new_tokens, seed):
        out = generator.generate(system, user, spec)
        final, report = assemble(out.text, inst)
        attempts.append(Attempt(i, spec, report, out.prompt_tokens, out.output_tokens))
        best = (final, report)
        if not should_retry(report):
            break
    return _finish(inst, best, attempts)


async def arun(payload: dict, generator: AsyncGenerator, *, retries: int | None = None,
               max_new_tokens: int | None = None, seed: int | None = None,
               instance: dict | None = None) -> PipelineResult:
    """Async twin of `run`, for the HTTP service.

    Deliberately a parallel loop rather than a sync/async bridge: the shared
    decisions live in `sampling_for_attempt` and `should_retry`, so the two
    cannot diverge on the part that matters.
    """
    retries = config.RETRIES if retries is None else retries
    max_new_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    inst = instance if instance is not None else prepare(payload)
    system, user = build_full_prompt(inst)

    attempts: list[Attempt] = []
    best: tuple[str, dict] | None = None
    for i, spec in _specs(retries, max_new_tokens, seed):
        out: GenerationOutput = await generator.agenerate(system, user, spec)
        final, report = assemble(out.text, inst)
        attempts.append(Attempt(i, spec, report, out.prompt_tokens, out.output_tokens))
        best = (final, report)
        if not should_retry(report):
            break
    return _finish(inst, best, attempts)


def meta(instance: dict) -> dict[str, Any]:
    """Identifying fields callers want alongside the markdown."""
    m = instance.get("meta", {})
    return {"commodity": m.get("commodity"), "report_title": m.get("report_title")}
