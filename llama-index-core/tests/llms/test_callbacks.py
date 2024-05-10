import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLMWithNonyieldingChatStream


@pytest.fixture()
def nonyielding_llm() -> LLM:
    return MockLLMWithNonyieldingChatStream()


@pytest.fixture()
def prompt() -> str:
    return "test prompt"


def test_llm_stream_chat_handles_nonyielding_stream(
    nonyielding_llm: LLM, prompt: str
) -> None:
    response = nonyielding_llm.stream_chat([ChatMessage(role="user", content=prompt)])
    for _ in response:
        pass


@pytest.mark.asyncio()
async def test_llm_astream_chat_handles_nonyielding_stream(
    nonyielding_llm: LLM, prompt: str
) -> None:
    response = await nonyielding_llm.astream_chat(
        [ChatMessage(role="user", content=prompt)]
    )
    async for _ in response:
        pass
