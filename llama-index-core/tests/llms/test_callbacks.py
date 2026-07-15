from typing import Any, Dict, List, Optional

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
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


class _SecretMockLLM(MockLLM):
    """MockLLM with a credential field, to assert it never reaches callbacks."""

    api_key: Optional[str] = None


class _CapturingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__([], [])
        self.serialized_payloads: List[Dict[str, Any]] = []

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if payload and EventPayload.SERIALIZED in payload:
            self.serialized_payloads.append(payload[EventPayload.SERIALIZED])
        return event_id

    def on_event_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_to_payload_excludes_secrets() -> None:
    llm = _SecretMockLLM()
    llm.api_key = "sk-super-secret"
    payload = llm.to_payload()
    assert "api_key" not in payload
    assert payload["model_name"] == llm.metadata.model_name
    assert payload["class_name"] == llm.class_name()


def test_callback_serialized_excludes_secrets(prompt: str) -> None:
    handler = _CapturingHandler()
    llm = _SecretMockLLM(callback_manager=CallbackManager([handler]))
    llm.api_key = "sk-super-secret"
    llm.complete(prompt)

    assert handler.serialized_payloads
    for serialized in handler.serialized_payloads:
        assert "api_key" not in serialized
        assert "sk-super-secret" not in str(serialized)
