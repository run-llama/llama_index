"""
Tests for ``_extract_reasoning_delta`` in ``llama_index.protocols.ag_ui.agent``.

These tests cover both LLM transports the AG-UI workflow can drive:

* OpenAI Responses API (``OpenAIResponses``) -- ``resp.raw`` is a
  ``ResponseStreamEvent`` whose ``type`` is
  ``response.reasoning_summary_text.delta`` and ``.delta`` carries the
  incremental summary text.
* OpenAI Chat Completions (``OpenAI``, DeepSeek, vLLM) -- ``resp.raw`` is a
  ``ChatCompletionChunk`` whose ``delta`` carries ``reasoning_content`` in
  either pydantic ``model_extra`` or as a plain attribute / dict key.

Reasoning extraction is strictly additive: any unrecognized shape returns
``""`` so the surrounding text-streaming path is unaffected.
"""

from types import SimpleNamespace

from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.protocols.ag_ui.agent import (
    _RESPONSES_REASONING_DELTA_TYPE,
    _extract_reasoning_delta,
)


def _mk_resp(raw):
    """Build a minimal ChatResponse with the given ``raw`` payload."""
    return ChatResponse(
        message=ChatMessage(role="assistant", content=""),
        delta="",
        raw=raw,
    )


# --- Responses API -------------------------------------------------------


def test_responses_api_object_with_reasoning_delta() -> None:
    raw = SimpleNamespace(
        type=_RESPONSES_REASONING_DELTA_TYPE,
        delta="step one",
    )
    assert _extract_reasoning_delta(_mk_resp(raw)) == "step one"


def test_responses_api_dict_with_reasoning_delta() -> None:
    raw = {"type": _RESPONSES_REASONING_DELTA_TYPE, "delta": "step two"}
    assert _extract_reasoning_delta(_mk_resp(raw)) == "step two"


def test_responses_api_unrelated_event_returns_empty() -> None:
    raw = SimpleNamespace(type="response.output_text.delta", delta="hi")
    assert _extract_reasoning_delta(_mk_resp(raw)) == ""


def test_responses_api_reasoning_shaped_unknown_type_returns_empty() -> None:
    """
    An SDK rename to a different reasoning-shaped type returns ``""``
    (and warns once via the logger), it must not raise.
    """
    raw = SimpleNamespace(type="response.reasoning_text.delta", delta="x")
    assert _extract_reasoning_delta(_mk_resp(raw)) == ""


# --- Chat Completions ----------------------------------------------------


def test_chat_completions_reasoning_content_model_extra() -> None:
    """OpenAI SDK pydantic chunk: non-standard fields land in ``model_extra``."""
    delta = SimpleNamespace(
        model_extra={"reasoning_content": "thinking..."},
        content=None,
    )
    raw = SimpleNamespace(
        type=None,
        choices=[SimpleNamespace(delta=delta)],
    )
    assert _extract_reasoning_delta(_mk_resp(raw)) == "thinking..."


def test_chat_completions_reasoning_content_direct_attr() -> None:
    """Some SDK builds expose ``reasoning_content`` directly on the delta."""
    delta = SimpleNamespace(reasoning_content="hmm", content=None)
    raw = SimpleNamespace(
        type=None,
        choices=[SimpleNamespace(delta=delta)],
    )
    assert _extract_reasoning_delta(_mk_resp(raw)) == "hmm"


def test_chat_completions_plain_dict() -> None:
    """A plain-dict raw chunk (e.g. when SDK has been deserialized through JSON)."""
    raw = {
        "type": None,
        "choices": [{"delta": {"reasoning_content": "ponder"}}],
    }
    assert _extract_reasoning_delta(_mk_resp(raw)) == "ponder"


def test_chat_completions_no_reasoning_returns_empty() -> None:
    delta = SimpleNamespace(model_extra={}, content="hello")
    raw = SimpleNamespace(
        type=None,
        choices=[SimpleNamespace(delta=delta)],
    )
    assert _extract_reasoning_delta(_mk_resp(raw)) == ""


# --- Edge cases ----------------------------------------------------------


def test_no_raw_returns_empty() -> None:
    assert _extract_reasoning_delta(_mk_resp(None)) == ""


def test_empty_choices_returns_empty() -> None:
    raw = SimpleNamespace(type=None, choices=[])
    assert _extract_reasoning_delta(_mk_resp(raw)) == ""
