from llama_index.bridge.langchain import FakeListLLM
from llama_index.llms.base import ChatMessage
from llama_index.llms.langchain import LangChainLLM


def test_basic() -> None:
    lc_llm = FakeListLLM(responses=["test response 1", "test response 2"])
    llm = LangChainLLM(llm=lc_llm)

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.chat([message])
