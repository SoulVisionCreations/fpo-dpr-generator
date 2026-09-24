"""The prompt the model sees must not drift.

`tests/golden/prompts.json` was captured from the pre-refactor code path. If a
change here fails, the refactor moved the model's input — which silently changes
output quality without breaking anything visibly. Regenerate the golden file only
when a prompt change is genuinely intended.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((Path(__file__).parent / "golden" / "prompts.json").read_text())


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.mark.parametrize("rel", sorted(GOLDEN))
def test_prompt_and_engine_numbers_are_unchanged(rel):
    from dpr import pipeline
    from dpr.authoring import build_full_prompt

    expected = GOLDEN[rel]
    inst = pipeline.prepare(json.loads((BASE / rel).read_text()))
    system, user = build_full_prompt(inst)

    assert _sha(system) == expected["sys_sha"], f"system prompt changed for {rel}"
    assert _sha(user) == expected["usr_sha"], f"user prompt changed for {rel}"
    assert inst["computed"]["project_cost"]["total_project_cost"] == expected["tpc"]
    assert inst["computed"]["dscr"]["average"] == expected["dscr"]


def test_system_prompt_is_payload_independent():
    """Every payload shares one SYSTEM string; a per-payload system prompt would
    be a bug, and would also defeat prefix caching in vLLM."""
    assert len({g["sys_sha"] for g in GOLDEN.values()}) == 1


def test_encode_produces_clean_prompt_ids():
    """The production encoder must return real token ids starting with one <bos>.

    Guards a bug this code already had: apply_chat_template can return a
    BatchEncoding, and a naive list() of it yields the dict's KEYS as "token ids".
    """
    from transformers import AutoTokenizer

    from dpr import config, pipeline
    from dpr.authoring import build_full_prompt

    adapter = config.DEFAULT_ADAPTER
    if not adapter.is_dir():
        pytest.skip("adapter not present")
    tok = AutoTokenizer.from_pretrained(str(adapter))

    inst = pipeline.prepare(json.loads((BASE / "data/examples/mango_juice.json").read_text()))
    system, user = build_full_prompt(inst)
    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    from serve.engine import VLLMGenerator

    # Exercise the real encoder, so this test guards the production path.
    ids = VLLMGenerator.encode(type("_", (), {"tok": tok})(), system, user)
    bos = tok.bos_token_id
    assert ids[0] == bos
    assert ids.count(bos) == 1, "templated ids should carry exactly one BOS"

    assert len(ids) > 500 and all(isinstance(i, int) for i in ids)

    # Document the current tokenizer's round-trip behaviour. add_bos_token=False
    # means re-tokenizing the rendered string is currently lossless; we still pass
    # ids so the service does not depend on that staying true.
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    assert tok(text)["input_ids"] == ids, (
        "string round-trip diverged from the templated ids — serve/engine.py "
        "passes ids so this is not fatal, but the tokenizer config changed"
    )


def test_stop_tokens_match_generation_config():
    """config.STOP_TOKEN_IDS must equal the model's real eos list.

    The tokenizer reports only <eos> (1), but the model ends its turn with
    <turn|> (106). A backend honouring only the tokenizer would generate to
    max_tokens on every request.
    """
    from dpr import config
    assert config.STOP_TOKEN_IDS == [1, 106, 50]
