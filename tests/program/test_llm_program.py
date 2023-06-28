"""Test LLM program."""

from llama_index.program.llm_program import LLMTextCompletionProgram
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.prompts.prompts import Prompt
from llama_index.llm_predictor.base import LLMPredictor
from unittest.mock import patch, MagicMock
from typing import Any, Tuple
from pydantic import BaseModel
from llama_index.bridge.langchain import BaseLanguageModel
import json


def mock_llmpredictor_predict(prompt: str, **prompt_args: Any) -> str:
    """Mock LLMPredictor predict."""
    test_object = {"hello": "world"}
    return json.dumps(test_object)


class TestModel(BaseModel):
    hello: str


@patch.object(BaseLanguageModel, "predict", side_effect=mock_llmpredictor_predict)
def test_llm_program(mock_predict: Any) -> None:
    """Test LLM program."""
    output_parser = PydanticOutputParser(output_cls=TestModel)
    llm_program = LLMTextCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt_template_str="This is a test prompt with a {test_input}.",
    )
    # mock llm
    llm_program._llm = MagicMock()
    llm_program._llm.predict = mock_llmpredictor_predict
    obj_output = llm_program(test_input="hello")
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"
