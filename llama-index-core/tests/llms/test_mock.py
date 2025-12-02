import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    TextBlock,
    DocumentBlock,
    ImageBlock,
)


@pytest.fixture()
def messages() -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="hello world"),
                DocumentBlock(data=b"hello world"),
                ImageBlock(image=b"1px"),
            ],
        )
    ]


def test_mock_llm_stream_complete_empty_prompt_no_max_tokens() -> None:
    """
    Test that MockLLM.stream_complete with an empty prompt and max_tokens=None
    does not raise a validation error.
    This test case is based on issue #19353.
    """
    llm = MockLLM(max_tokens=None)
    response_gen = llm.stream_complete("")

    # Consume the generator to trigger the potential error
    responses = list(response_gen)

    # Check that we received a single, empty response
    assert len(responses) == 1
    assert responses[0].text == ""
    assert responses[0].delta == ""


def test_mock_function_calling_llm_init() -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    assert llm.max_tokens == 200
    assert llm.metadata.is_function_calling_model


def test_mock_function_calling_llm_sync_methods(messages: list[ChatMessage]) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    result = llm.chat(messages)
    assert (
        result.message.content
        == "hello world<document>hello world</document><image>1px</image>"
    )
    cont = ""
    stream = llm.stream_chat(messages)
    for s in stream:
        cont += s.message.content or ""
    assert cont == "hello world<document>hello world</document><image>1px</image>"


@pytest.mark.asyncio
async def test_mock_function_calling_llm_async_methods(
    messages: list[ChatMessage],
) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    result = await llm.achat(messages)
    assert (
        result.message.content
        == "hello world<document>hello world</document><image>1px</image>"
    )
    cont = ""
    stream = await llm.astream_chat(messages)
    async for s in stream:
        cont += s.message.content or ""
    assert cont == "hello world<document>hello world</document><image>1px</image>"
