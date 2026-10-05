"""Request/response models for the DPR service."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dpr import config


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retries: int = Field(default=config.RETRIES, ge=0, le=5)
    max_new_tokens: int = Field(default=config.MAX_NEW_TOKENS, ge=256, le=8192)
    seed: int | None = None
    strict: bool = Field(default=False, description="unclean verification -> 422 instead of 200")
    require_complete_inputs: bool = Field(
        default=False,
        description=("Reject rather than substitute. If the payload omits any financial "
                     "field, return 422 listing them instead of using a line default. "
                     "Use this for bank/NABARD submissions, where every figure must "
                     "originate from the applicant."))
    include_instance: bool = Field(default=False, description="return the computed engine instance")


class GenerateRequest(BaseModel):
    # `extra="forbid"` guards the ENVELOPE only. `payload` is deliberately an
    # opaque dict: convert_website accepts shapes a strict model would reject, and
    # the API must accept exactly what --payload accepts.
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    options: Options = Field(default_factory=Options)

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_form(cls, data: Any) -> Any:
        """Accept the form payload with or without the envelope.

        Callers that post the FPO Document Builder form directly — a gateway
        forwarding the client's body unchanged, or anyone using the form's own
        shape — send {"document_type": ..., "fields": {...}} at the top level.
        Wrap that rather than rejecting it; `extra="forbid"` would otherwise 422
        on every form field.
        """
        if isinstance(data, dict) and "payload" not in data:
            return {"payload": data}
        return data


class Verification(BaseModel):
    """The assemble() report, verbatim — all four keys."""
    ok: bool
    unknown_placeholders: list[str] = []
    unplaced_sections: list[str] = []
    number_issues: list[dict[str, Any]] = []


class Meta(BaseModel):
    commodity: str | None = None
    report_title: str | None = None
    attempts: int = 0
    clean: bool = False
    total_project_cost_lakh: float | None = None
    average_dscr: float | None = None
    pat_year1: float | None = None
    balances: bool | None = None
    financially_viable: bool | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    generation_ms: int = 0
    model: str | None = None
    model_revision: str | None = None
    adapter: str | None = None


class DefaultApplied(BaseModel):
    """A figure the FPO did not supply, substituted by the deterministic converter.

    Never produced by the language model. Machine-readable so a caller can assert
    the list is empty before sending a DPR to a bank.
    """
    field: str
    value: Any
    unit: str = ""
    source: str = ""


class GenerateResponse(BaseModel):
    request_id: str
    markdown: str
    verification: Verification
    meta: Meta
    warnings: list[str] = []
    defaults_applied: list[DefaultApplied] = []
    instance: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
