"""LLM predictor tests."""

import pytest
from gpt_index.llm_predictor.structured import StructuredLLMPredictor, LLMPredictor
from gpt_index.prompts.prompts import Prompt, SimpleInputPrompt
from typing import Any, Tuple
from unittest.mock import patch
from gpt_index.output_parsers.base import BaseOutputParser


class MockOutputParser(BaseOutputParser):
    """Mock output parser."""

    def parse(self, output: str) -> str:
        """Parse output."""
        return output + "\n" + output

    def format(self, output: str) -> str:
        """Format output."""
        return output


def mock_llmpredictor_predict(prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
    """Mock LLMPredictor predict."""
    return prompt_args["query_str"], "mocked formatted prompt"


@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
def test_struct_llm_predictor(mock_predict: Any):
    """Test LLM predictor."""
    llm_predictor = StructuredLLMPredictor()
    output_parser = MockOutputParser()
    prompt = SimpleInputPrompt("{query_str}", output_parser=output_parser)
    llm_prediction, formatted_output = llm_predictor.predict(
        prompt, query_str="hello world"
    )
    assert llm_prediction == "hello world\nhello world"

    # no change
    prompt = SimpleInputPrompt("{query_str}")
    llm_prediction, formatted_output = llm_predictor.predict(
        prompt, query_str="hello world"
    )
    assert llm_prediction == "hello world"
