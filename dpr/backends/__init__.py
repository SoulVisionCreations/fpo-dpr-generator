"""Narrative-lane backends: HF transformers (CLI) and vLLM (service)."""
from dpr.backends.base import (
    AdapterNotFoundError,
    AsyncGenerator,
    GenerationOutput,
    Generator,
    SamplingSpec,
)

__all__ = ["AdapterNotFoundError", "AsyncGenerator", "GenerationOutput",
           "Generator", "SamplingSpec"]
