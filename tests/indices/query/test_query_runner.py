"""Test query runner."""

from typing import Any
from unittest.mock import patch

from gpt_index import PromptHelper
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMChain,
    LLMMetadata,
    LLMPredictor,
)
from gpt_index.readers.schema.base import Document


def mock_llmchain_predict(**full_prompt_args: Any) -> str:
    """Mock LLMChain predict with a generic response."""
    return "foo bar 2"


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.langchain_helpers.chain_wrapper.OpenAI")
@patch.object(LLMPredictor, "get_llm_metadata", return_value=LLMMetadata())
@patch.object(LLMChain, "__init__", return_value=None)
def test_passing_args_to_query(
    _mock_init: Any,
    _mock_llm_metadata: Any,
    _mock_openai: Any,
    _mock_predict: Any,
) -> None:
    """Test passing args to query works.

    Test that passing LLMPredictor from build index to query works.

    """
    doc_text = "Hello world."
    doc = Document(doc_text)
    llm_predictor = LLMPredictor()
    prompt_helper = PromptHelper.from_llm_predictor(llm_predictor)
    # index construction should not use llm_predictor at all
    index = GPTListIndex(
        [doc], llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    # should use llm_predictor during query time
    response = index.query("What is?")
    assert str(response) == "foo bar 2"
