import pytest
from llama_index.llms.vertex import Vertex
from llama_index.llms.base import ChatMessage,MessageRole,CompletionResponse

def test_vertex_initialization() -> None:
    llm = Vertex()
    assert llm.class_name() == "Vertex_LLM"
    assert llm.model == llm._client._model_id


def test_vertex_call() -> None:
    llm = Vertex(temperature=0)
    output = llm.complete("Say foo:")
    assert isinstance(output.text, str)

@pytest.mark.scheduled
def test_vertex_generate() -> None:
    llm = Vertex(model="text-bison")
    output = llm.complete("hello",temperature=0.4,candidate_count=2)
    assert isinstance(output, CompletionResponse)
    assert len(output.raw['candidates']) == 2


@pytest.mark.scheduled
def test_vertex_generate_code() -> None:
    llm = Vertex(model="code-bison")
    output = llm.complete("generate a python method that says foo:",temperature = 0.4)
    assert isinstance(output, CompletionResponse)
    


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_vertex_agenerate() -> None:
    llm = Vertex(model="text-bison")
    output = await llm.acomplete("Please say foo:")
    assert isinstance(output, CompletionResponse)


@pytest.mark.scheduled
def test_vertex_stream() -> None:
    llm = Vertex()
    outputs = list(llm.stream_complete("Please say foo:"))
    assert isinstance(outputs[0].text, str)


@pytest.mark.asyncio
async def test_vertex_consistency() -> None:
    llm = Vertex(temperature= 0)
    output = llm.complete("Please say foo:")
    streaming_output = list(llm.stream_complete("Please say foo:"))
    async_output = await llm.acomplete("Please say foo:")
    assert output.text == streaming_output[-1].text
    assert output.text == async_output.text
