"""Backend protocol for the narrative lane.

The pipeline owns the retry/assemble/verify loop; a backend owns only "turn a
(system, user) prompt into text". Keeping that seam explicit is what lets the CLI
run HF transformers and the HTTP service run vLLM over one copy of the logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


class AdapterNotFoundError(FileNotFoundError):
    """The LoRA adapter directory is missing.

    A plain exception rather than sys.exit: SystemExit is a BaseException, so a
    server's `except Exception` would not catch it and the worker would die.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        super().__init__(f"adapter not found: {path}")


@dataclass(frozen=True)
class GenerationOutput:
    """One completion plus the token accounting callers need for logging."""
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    # Top-k logprobs of the FIRST generated token, when requested. Used to prove
    # a LoRA is actually applied: identical argmax is common, identical
    # probability mass is not.
    first_logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SamplingSpec:
    """Backend-neutral sampling settings for a single attempt.

    Attempt 0 is greedy; retries sample. Expressed as data so the sync and async
    loops cannot drift, and so tests can assert what each attempt asked for
    without a model.
    """
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@runtime_checkable
class Generator(Protocol):
    """Sync backend (the CLI's HF path)."""

    def generate(self, system: str, user: str, spec: SamplingSpec) -> GenerationOutput: ...


@runtime_checkable
class AsyncGenerator(Protocol):
    """Async backend (the service's vLLM path)."""

    async def agenerate(self, system: str, user: str, spec: SamplingSpec) -> GenerationOutput: ...
