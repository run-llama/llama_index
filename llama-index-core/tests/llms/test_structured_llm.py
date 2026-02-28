"""Tests for StructuredLLM output handling.

Regression tests for https://github.com/run-llama/llama_index/issues/16604
"""

from unittest.mock import AsyncMock, patch

import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import MockLLM
from llama_index.core.llms.structured_llm import StructuredLLM


class SimpleOutput(BaseModel):
    answer: str


def test_structured_llm_parses_string_output() -> None:
    """StructuredLLM.chat should parse a valid JSON string into the model."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput)

    with patch.object(
        type(mock_llm), "structured_predict", return_value='{"answer": "42"}'
    ):
        response = sllm.chat(
            [ChatMessage(role="user", content="what is the answer?")]
        )

    assert isinstance(response.raw, SimpleOutput)
    assert response.raw.answer == "42"


def test_structured_llm_retries_on_invalid_output() -> None:
    """StructuredLLM should retry when the first attempt returns garbage."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput, max_retries=1)

    # First call returns garbage, second returns valid JSON
    with patch.object(
        type(mock_llm),
        "structured_predict",
        side_effect=["not valid json at all", '{"answer": "ok"}'],
    ):
        response = sllm.chat(
            [ChatMessage(role="user", content="test")]
        )

    assert isinstance(response.raw, SimpleOutput)
    assert response.raw.answer == "ok"


def test_structured_llm_raises_after_exhausted_retries() -> None:
    """StructuredLLM should raise ValueError when all retries are exhausted."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput, max_retries=1)

    with patch.object(
        type(mock_llm), "structured_predict", return_value="garbage"
    ):
        with pytest.raises(ValueError, match="failed to produce valid"):
            sllm.chat([ChatMessage(role="user", content="test")])


def test_structured_llm_passes_model_output_through() -> None:
    """When structured_predict returns a proper model, no retry needed."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput)

    with patch.object(
        type(mock_llm),
        "structured_predict",
        return_value=SimpleOutput(answer="direct"),
    ):
        response = sllm.chat(
            [ChatMessage(role="user", content="test")]
        )

    assert response.raw.answer == "direct"


@pytest.mark.asyncio()
async def test_structured_llm_async_parses_string() -> None:
    """StructuredLLM.achat should parse a valid JSON string."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput)

    with patch.object(
        type(mock_llm),
        "astructured_predict",
        new_callable=AsyncMock,
        return_value='{"answer": "hello"}',
    ):
        response = await sllm.achat(
            [ChatMessage(role="user", content="say hello")]
        )

    assert isinstance(response.raw, SimpleOutput)
    assert response.raw.answer == "hello"


@pytest.mark.asyncio()
async def test_structured_llm_async_retries_on_invalid() -> None:
    """StructuredLLM.achat should retry on invalid output."""
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput, max_retries=1)

    with patch.object(
        type(mock_llm),
        "astructured_predict",
        new_callable=AsyncMock,
        side_effect=["bad output", '{"answer": "fixed"}'],
    ):
        response = await sllm.achat(
            [ChatMessage(role="user", content="test")]
        )

    assert response.raw.answer == "fixed"
