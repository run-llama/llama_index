"""Test LLM program."""

import json
from unittest.mock import MagicMock

from pydantic.v1 import BaseModel

from llama_index.llms.base import CompletionResponse
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.program.llm_program import LLMTextCompletionProgram


class MockLLM(MagicMock):
    def complete(self, prompt: str) -> CompletionResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return CompletionResponse(text=text)


class TestModel(BaseModel):
    hello: str


def test_llm_program() -> None:
    """Test LLM program."""
    output_parser = PydanticOutputParser(output_cls=TestModel)
    llm_program = LLMTextCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt_template_str="This is a test prompt with a {test_input}.",
        llm=MockLLM(),
    )
    # mock llm
    obj_output = llm_program(test_input="hello")
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"
