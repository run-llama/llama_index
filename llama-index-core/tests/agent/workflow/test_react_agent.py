from typing import Any, AsyncGenerator, List
import pytest
from pydantic import BaseModel

from llama_index.core.agent.workflow import ReActAgent, safe_model_dump
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.prompts import PromptTemplate


def test_react_agent_prompts():
    llm = MockLLM()
    agent = ReActAgent(
        llm=llm,
        tools=[],
    )

    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert isinstance(prompts["react_header"], PromptTemplate)

    new_prompt = "New prompt"
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt in str(prompts["react_header"])

    new_prompt = PromptTemplate("New prompt 2")
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt == prompts["react_header"]


class MockDictResponse(dict):
    """Simulates response objects like DashScopeResponse that raise KeyError in __getattr__."""

    def __getattr__(self, name: str) -> Any:
        return self[name]


class DummyPydanticModel(BaseModel):
    key: str = "value"


def test_safe_model_dump():
    # BaseModel dumped to dict
    model = DummyPydanticModel(key="test")
    assert safe_model_dump(model) == {"key": "test"}

    # Dict and primitives returned as-is
    assert safe_model_dump({"a": 1}) == {"a": 1}
    assert safe_model_dump("string") == "string"
    assert safe_model_dump(None) is None

    # Object whose __getattr__ raises KeyError (e.g., DashScopeResponse)
    mock_raw = MockDictResponse({"status": "ok"})
    # Normal access works
    assert mock_raw["status"] == "ok"
    # Accessing non-existent attribute raises KeyError
    with pytest.raises(KeyError):
        _ = mock_raw.non_existent_key
    # safe_model_dump handles it gracefully without raising KeyError
    assert safe_model_dump(mock_raw) is mock_raw

    # Object with metaclass raising exception on instance check
    class BadMeta(type):
        def __instancecheck__(cls, instance: Any) -> bool:
            raise KeyError("__pydantic_validator__")

    class BadClass(metaclass=BadMeta):
        pass

    assert safe_model_dump(BadClass()) is not None


class DashScopeLikeLLM(MockFunctionCallingLLM):
    """Mock LLM returning ChatResponse with dict-like raw response raising KeyError on missing attrs."""

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raw_obj = MockDictResponse({"response_id": "test_123"})

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Thought: Done\nAnswer: 42",
                ),
                raw=raw_obj,
            )

        return gen()

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        raw_obj = MockDictResponse({"response_id": "test_123"})
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Thought: Done\nAnswer: 42",
            ),
            raw=raw_obj,
        )


@pytest.mark.asyncio
async def test_react_agent_with_dict_like_raw_response():
    """Verify ReActAgent handles LLM responses whose raw object raises KeyError in __getattr__ (#18694)."""
    agent = ReActAgent(
        llm=DashScopeLikeLLM(),
        tools=[],
    )
    handler = agent.run("What is the answer?")
    async for _ in handler.stream_events():
        pass
    response = await handler
    assert "42" in str(response)
    assert response.raw["response_id"] == "test_123"
