"""Test LLM program."""

import json
from typing import Sequence
from unittest.mock import MagicMock

from llama_index.bridge.pydantic import BaseModel
from llama_index.core.llms.types import (
    CompletionResponse,
)
from llama_index.multi_modal_llms import MultiModalLLMMetadata
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.schema import ImageDocument


class MockMultiModalLLM(MagicMock):
    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument]
    ) -> CompletionResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return CompletionResponse(text=text)

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata()


class TestModel(BaseModel):
    __test__ = False
    hello: str


def test_multi_modal_llm_program() -> None:
    """Test Multi Modal LLM Pydantic program."""
    output_parser = PydanticOutputParser(output_cls=TestModel)
    multi_modal_llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt_template_str="This is a test prompt with a {test_input}.",
        multi_modal_llm=MockMultiModalLLM(),
        image_documents=[ImageDocument()],
    )
    # mock Multi Modal llm
    obj_output = multi_modal_llm_program(test_input="hello")
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"
