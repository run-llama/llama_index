import pytest
from llama_index.llms.types import ChatMessage, CompletionResponse
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
    # complete
    llm = Vertex(temperature=0, model="gemini-pro")
    output = llm.complete("Say foo:")
    assert "foo" in output.text.lower()
    streaming_output = list(llm.stream_complete("Please say foo:"))
    assert "foo" in streaming_output[-1].text

    async_output = await llm.acomplete("Please say foo:")
    assert "foo" in async_output.text

    # chat
    history = [
        ChatMessage(role="user", content="Say foo:"),
        ChatMessage(role="assistant", content="Foo with love !"),
        ChatMessage(role="user", content="Please repeat"),
    ]
    output = llm.chat(history)
    assert "Foo with love !" in output.message.content
    streaming_output = list(llm.stream_chat(history))
    assert "Foo with love !" in streaming_output[-1].message.content
    async_output = await llm.achat(history)
    assert "Foo with love !" in async_output.message.content
