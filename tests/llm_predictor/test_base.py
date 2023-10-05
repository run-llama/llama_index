"""LLM predictor tests."""
from typing import Any
from unittest.mock import patch

from llama_index.llm_predictor.structured import LLMPredictor, StructuredLLMPredictor
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.base import PromptTemplate
from llama_index.types import BaseOutputParser


class MockOutputParser(BaseOutputParser):
    """Mock output parser."""

    def parse(self, output: str) -> str:
        """Parse output."""
        return output + "\n" + output

    def format(self, output: str) -> str:
        """Format output."""
        return output


def mock_llmpredictor_predict(prompt: BasePromptTemplate, **prompt_args: Any) -> str:
    """Mock LLMPredictor predict."""
    return prompt_args["query_str"]


@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_struct_llm_predictor(mock_init: Any, mock_predict: Any) -> None:
    """Test LLM predictor."""
    llm_predictor = StructuredLLMPredictor()
    output_parser = MockOutputParser()
    prompt = PromptTemplate("{query_str}", output_parser=output_parser)
    llm_prediction = llm_predictor.predict(prompt, query_str="hello world")
    assert llm_prediction == "hello world\nhello world"

    # no change
    prompt = PromptTemplate("{query_str}")
    llm_prediction = llm_predictor.predict(prompt, query_str="hello world")
    assert llm_prediction == "hello world"
