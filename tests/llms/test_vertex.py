from typing import Sequence

import pytest
from llama_index.core.llms.types import ChatMessage, CompletionResponse
from llama_index.llms.vertex import Vertex
from llama_index.llms.vertex_utils import init_vertexai

try:
    init_vertexai()
    vertex_init = True
except Exception as e:
    vertex_init = False


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
def test_vertex_initialization() -> None:
    llm = Vertex()
    assert llm.class_name() == "Vertex"
    assert llm.model == llm._client._model_id


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
def test_vertex_call() -> None:
    llm = Vertex(temperature=0)
    output = llm.complete("Say foo:")
    assert isinstance(output.text, str)


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
def test_vertex_generate() -> None:
    llm = Vertex(model="text-bison")
    output = llm.complete("hello", temperature=0.4, candidate_count=2)
    assert isinstance(output, CompletionResponse)


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
def test_vertex_generate_code() -> None:
    llm = Vertex(model="code-bison")
    output = llm.complete("generate a python method that says foo:", temperature=0.4)
    assert isinstance(output, CompletionResponse)


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
@pytest.mark.asyncio()
async def test_vertex_agenerate() -> None:
    llm = Vertex(model="text-bison")
    output = await llm.acomplete("Please say foo:")
    assert isinstance(output, CompletionResponse)


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
def test_vertex_stream() -> None:
    llm = Vertex()
    outputs = list(llm.stream_complete("Please say foo:"))
    assert isinstance(outputs[0].text, str)


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
@pytest.mark.asyncio()
async def test_vertex_consistency() -> None:
    llm = Vertex(temperature=0)
    output = llm.complete("Please say foo:")
    streaming_output = list(llm.stream_complete("Please say foo:"))
    async_output = await llm.acomplete("Please say foo:")
    assert output.text == streaming_output[-1].text
    assert output.text == async_output.text


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
@pytest.mark.asyncio()
async def test_vertex_gemini_call() -> None:
    llm = Vertex(temperature=0, model="gemini-pro")
    output = llm.complete("Say foo:")
    assert "foo" in output.text.lower()
    streaming_output = list(llm.stream_complete("Please say foo:"))
    assert "foo" in streaming_output[-1].text

    async_output = await llm.acomplete("Please say foo:")
    assert "foo" in async_output.text

    history = [
        ChatMessage(role="user", content="Say foo:"),
        ChatMessage(role="assistant", content="Foo with love !"),
        ChatMessage(role="user", content="Please repeat"),
    ]
    await _call_chat_and_assert(llm, history, "foo with love !")


@pytest.mark.skipif(vertex_init is False, reason="vertex not installed")
@pytest.mark.asyncio()
async def test_vertex_gemini_vision_call() -> None:
    llm = Vertex(temperature=0, model="gemini-pro-vision")
    output = llm.complete("Say foo:")
    assert "foo" in output.text.lower()
    streaming_output = list(llm.stream_complete("Please say foo:"))
    assert "foo" in streaming_output[-1].text
    async_output = await llm.acomplete("Please say foo:")
    assert "foo" in async_output.text

    history = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Explain what is in the image below:"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,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",
                },
            ],
        ),
    ]
    await _call_chat_and_assert(llm, history, "espresso")


async def _call_chat_and_assert(
    llm: Vertex, history: Sequence[ChatMessage], expected_lower_message: str
) -> None:
    output = llm.chat(history)
    assert expected_lower_message in output.message.content.lower()
    streaming_output = list(llm.stream_chat(history))
    assert expected_lower_message in streaming_output[-1].message.content.lower()
    async_output = await llm.achat(history)
    assert expected_lower_message in async_output.message.content.lower()
