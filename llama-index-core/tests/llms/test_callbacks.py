from typing import Any, List, Optional

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM
from llama_index.core.llms.mock import MockLLMWithNonyieldingChatStream


@pytest.fixture()
def nonyielding_llm() -> LLM:
    return MockLLMWithNonyieldingChatStream()


@pytest.fixture()
def llm() -> LLM:
    return MockLLM()


@pytest.fixture()
def prompt() -> str:
    return "test prompt"


def test_llm_stream_chat_handles_nonyielding_stream(
    nonyielding_llm: LLM, prompt: str
) -> None:
    response = nonyielding_llm.stream_chat([ChatMessage(role="user", content=prompt)])
    for _ in response:
        pass


@pytest.mark.asyncio
async def test_llm_astream_chat_handles_nonyielding_stream(
    nonyielding_llm: LLM, prompt: str
) -> None:
    response = await nonyielding_llm.astream_chat(
        [ChatMessage(role="user", content=prompt)]
    )
    async for _ in response:
        pass


def test_llm_complete_prompt_arg(llm: LLM, prompt: str) -> None:
    res = llm.complete(prompt)
    expected_res_text = prompt
    assert res.text == expected_res_text


def test_llm_complete_prompt_kwarg(llm: LLM, prompt: str) -> None:
    res = llm.complete(prompt=prompt)
    expected_res_text = prompt
    assert res.text == expected_res_text


def test_llm_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str) -> None:
    with pytest.raises(TypeError):
        llm.complete(prompt, prompt=prompt)


def test_llm_complete_throws_if_no_prompt(llm: LLM) -> None:
    with pytest.raises(ValueError):
        llm.complete()


def test_llm_stream_complete_prompt_arg(llm: LLM, prompt: str) -> None:
    res_text = "".join(r.delta for r in llm.stream_complete(prompt))
    expected_res_text = prompt
    assert res_text == expected_res_text


def test_llm_stream_complete_prompt_kwarg(llm: LLM, prompt: str) -> None:
    res_text = "".join(r.delta for r in llm.stream_complete(prompt=prompt))
    expected_res_text = prompt
    assert res_text == expected_res_text


def test_llm_stream_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str) -> None:
    with pytest.raises(TypeError):
        llm.stream_complete(prompt, prompt=prompt)


def test_llm_stream_complete_throws_if_no_prompt(llm: LLM) -> None:
    with pytest.raises(ValueError):
        llm.stream_complete()


class _KeyedMockLLM(MockLLM):
    api_key: Optional[str] = "sk-secret-should-not-leak"


class _SerializedCaptureHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.serialized: List[dict] = []

    def on_event_start(
        self, event_type, payload=None, event_id="", parent_id="", **kwargs: Any
    ) -> str:
        if payload and EventPayload.SERIALIZED in payload:
            self.serialized.append(payload[EventPayload.SERIALIZED])
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs: Any) -> None:
        ...

    def start_trace(self, trace_id=None) -> None:
        ...

    def end_trace(self, trace_id=None, trace_map=None) -> None:
        ...


# Regression test: the EventPayload.SERIALIZED callback payload must not
# contain api_key. It mirrors the model_dict the instrumentation event already
# redacts; leaking it exposes the credential to every callback handler.
def test_llm_callback_serialized_payload_redacts_api_key() -> None:
    handler = _SerializedCaptureHandler()
    llm = _KeyedMockLLM(callback_manager=CallbackManager([handler]))

    llm.complete("hello")
    llm.chat([ChatMessage(role="user", content="hello")])

    assert handler.serialized, "no SERIALIZED payload was captured"
    for serialized in handler.serialized:
        assert "api_key" not in serialized
        # other model config is still present (we did not over-strip)
        assert serialized.get("class_name") == llm.class_name()
