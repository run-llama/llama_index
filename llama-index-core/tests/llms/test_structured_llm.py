"""Tests for StructuredLLM string output handling.

Regression test for https://github.com/run-llama/llama_index/issues/16604
"""

from unittest.mock import patch

import pytest

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import MockLLM
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.base.llms.types import ChatMessage


class SimpleOutput(BaseModel):
    answer: str


def test_structured_llm_handles_string_output() -> None:
    """Test that StructuredLLM.chat handles string output from structured_predict.

    When the underlying LLM returns a JSON string instead of a Pydantic model,
    StructuredLLM should parse it into the expected model instead of crashing
    with AttributeError: 'str' object has no attribute 'model_dump_json'.
    """
    mock_llm = MockLLM()
    sllm = StructuredLLM(llm=mock_llm, output_cls=SimpleOutput)

    # Patch structured_predict on the LLM class to return a string
    with patch.object(
        type(mock_llm), "structured_predict", return_value='{"answer": "42"}'
    ):
        response = sllm.chat(
            [ChatMessage(role="user", content="what is the answer?")]
        )

    # Should succeed and return valid response (not crash with AttributeError)
    assert response.raw is not None
    assert isinstance(response.raw, SimpleOutput)
    assert response.raw.answer == "42"


@pytest.mark.asyncio()
async def test_structured_llm_handles_string_output_async() -> None:
    """Test that StructuredLLM.achat handles string output."""
    from unittest.mock import AsyncMock

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

    assert response.raw is not None
    assert isinstance(response.raw, SimpleOutput)
    assert response.raw.answer == "hello"

