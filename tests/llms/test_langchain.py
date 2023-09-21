import pytest
from typing import List

from llama_index.bridge.langchain import (
    FakeListLLM,
    SystemMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage,
    BaseMessage,
    Cohere,
    ChatOpenAI,
    OpenAI,
)

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.langchain_utils import from_lc_messages, to_lc_messages


def test_basic() -> None:
    lc_llm = FakeListLLM(responses=["test response 1", "test response 2"])
    llm = LangChainLLM(llm=lc_llm)

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.chat([message])


def test_to_lc_messages() -> None:
    lc_messages: List[BaseMessage] = [
        SystemMessage(content="test system message"),
        HumanMessage(content="test human message"),
        AIMessage(content="test ai message"),
        FunctionMessage(content="test function message", name="test function"),
    ]

    messages = from_lc_messages(lc_messages)

    for i in range(len(messages)):
        assert messages[i].content == lc_messages[i].content


def test_from_lc_messages() -> None:
    messages = [
        ChatMessage(content="test system message", role=MessageRole.SYSTEM),
        ChatMessage(content="test human message", role=MessageRole.USER),
        ChatMessage(content="test ai message", role=MessageRole.ASSISTANT),
        ChatMessage(
            content="test function message",
            role=MessageRole.FUNCTION,
            additional_kwargs={"name": "test function"},
        ),
    ]

    lc_messages = to_lc_messages(messages)

    for i in range(len(messages)):
        assert messages[i].content == lc_messages[i].content


try:
    import cohere  # noqa: F401
except ImportError:
    cohere = None  # type: ignore


@pytest.mark.skipif(cohere is None, reason="cohere not installed")
def test_metadata_sets_model_name() -> None:
    chat_gpt = LangChainLLM(
        llm=ChatOpenAI(model="gpt-4-0613", openai_api_key="model-name-tests")
    )
    assert chat_gpt.metadata.model_name == "gpt-4-0613"

    gpt35 = LangChainLLM(
        llm=OpenAI(model="gpt-3.5-turbo-0613", openai_api_key="model-name-tests")
    )
    assert gpt35.metadata.model_name == "gpt-3.5-turbo-0613"

    cohere_llm = LangChainLLM(
        llm=Cohere(model="j2-jumbo-instruct", cohere_api_key="XXXXXXX")
    )
    assert cohere_llm.metadata.model_name == "j2-jumbo-instruct"
