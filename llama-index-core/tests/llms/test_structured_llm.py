"""
Tests for StructuredLLM - validates output type before calling model_dump_json.

Regression tests for
"""

from typing import Any, Dict, Optional, Sequence, Type

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.types import Model


class TestOutput(BaseModel):
    """Test output model."""

    name: str = Field(description="A name")
    value: int = Field(description="A value")


class MockLLMReturnsModel(LLM):
    """Mock LLM that returns a proper Pydantic model from structured_predict."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=False)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="mock")
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text="mock")

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="mock")
        )

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text="mock")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls(name="test", value=42)

    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls(name="test_async", value=99)


class MockLLMReturnsString(MockLLMReturnsModel):
    """Mock LLM that returns a string instead of a Pydantic model."""

    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        # Simulate the bug: return a raw string instead of a model
        return "This is not a Pydantic model"  # type: ignore

    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return "This is not a Pydantic model"  # type: ignore


def test_structured_llm_chat_success() -> None:
    """Test that StructuredLLM.chat works correctly with valid model output."""
    llm = MockLLMReturnsModel()
    structured_llm = StructuredLLM(llm=llm, output_cls=TestOutput)

    result = structured_llm.chat(
        [ChatMessage(role=MessageRole.USER, content="give me a test")]
    )
    assert result.raw is not None
    assert isinstance(result.raw, TestOutput)
    assert result.raw.name == "test"
    assert result.raw.value == 42
    # content should be valid JSON
    assert '"name"' in result.message.content
    assert '"test"' in result.message.content


@pytest.mark.asyncio
async def test_structured_llm_achat_success() -> None:
    """Test that StructuredLLM.achat works correctly with valid model output."""
    llm = MockLLMReturnsModel()
    structured_llm = StructuredLLM(llm=llm, output_cls=TestOutput)

    result = await structured_llm.achat(
        [ChatMessage(role=MessageRole.USER, content="give me a test")]
    )
    assert result.raw is not None
    assert isinstance(result.raw, TestOutput)
    assert result.raw.name == "test_async"
    assert result.raw.value == 99


def test_structured_llm_chat_raises_on_string_output() -> None:
    """
    Test that StructuredLLM.chat raises a clear ValueError when
    structured_predict returns a string instead of a Pydantic model.

    This is the exact bug from issue #16604 where users see:
    AttributeError: 'str' object has no attribute 'model_dump_json'

    """
    llm = MockLLMReturnsString()
    structured_llm = StructuredLLM(llm=llm, output_cls=TestOutput)

    with pytest.raises(ValueError, match="expected a TestOutput instance"):
        structured_llm.chat(
            [ChatMessage(role=MessageRole.USER, content="give me a test")]
        )


@pytest.mark.asyncio
async def test_structured_llm_achat_raises_on_string_output() -> None:
    """
    Test async: StructuredLLM.achat raises a clear ValueError when
    astructured_predict returns a string.

    """
    llm = MockLLMReturnsString()
    structured_llm = StructuredLLM(llm=llm, output_cls=TestOutput)

    with pytest.raises(ValueError, match="expected a TestOutput instance"):
        await structured_llm.achat(
            [ChatMessage(role=MessageRole.USER, content="give me a test")]
        )


def test_structured_llm_error_message_is_descriptive() -> None:
    """
    Test that the error message includes the actual type and value received,
    helping users debug what went wrong.

    """
    llm = MockLLMReturnsString()
    structured_llm = StructuredLLM(llm=llm, output_cls=TestOutput)

    with pytest.raises(ValueError) as exc_info:
        structured_llm.chat(
            [ChatMessage(role=MessageRole.USER, content="give me a test")]
        )

    error_msg = str(exc_info.value)
    # Should mention the expected type
    assert "TestOutput" in error_msg
    # Should mention what was actually received
    assert "str" in error_msg
    # Should include the actual value for debugging
    assert "not a Pydantic model" in error_msg
