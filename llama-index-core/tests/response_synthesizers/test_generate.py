import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.generation import Generation


def test_synthesize() -> None:
    synthesizer = Generation(llm=MockLLM())
    response = synthesizer.synthesize(query="test", nodes=[])
    assert str(response) == "test"


def test_synthesize_stream() -> None:
    synthesizer = Generation(llm=MockLLM(), streaming=True)
    response = synthesizer.synthesize(query="test", nodes=[])

    gold = "test"
    i = 0
    for chunk in response.response_gen:
        assert chunk == gold[i]
        i += 1


@pytest.mark.asyncio()
async def test_asynthesize() -> None:
    synthesizer = Generation(llm=MockLLM())
    response = await synthesizer.asynthesize(query="test", nodes=[])
    assert str(response) == "test"


@pytest.mark.asyncio()
async def test_asynthesize_stream() -> None:
    synthesizer = Generation(llm=MockLLM(), streaming=True)
    response = await synthesizer.asynthesize(query="test", nodes=[])

    gold = "test"
    i = 0
    async for chunk in response.async_response_gen():
        assert chunk == gold[i]
        i += 1
