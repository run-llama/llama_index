"""Test LLM program."""

import json
import pytest
from typing import Sequence
from unittest.mock import MagicMock

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import LLMMetadata
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.llms import ImageBlock, ChatResponse, ChatMessage


@pytest.fixture()
def image_url() -> str:
    return "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"


class MagicLLM(MagicMock):
    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return ChatResponse(message=ChatMessage(role="assistant", content=text))

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()


class TestModel(BaseModel):
    __test__ = False
    hello: str


def test_multi_modal_llm_program(image_url: str) -> None:
    """Test Multi Modal LLM Pydantic program."""
    output_parser = PydanticOutputParser(output_cls=TestModel)
    multi_modal_llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt_template_str="This is a test prompt with a {test_input}.",
        multi_modal_llm=MagicLLM(),
        image_documents=[ImageBlock(url=image_url)],
    )
    # mock Multi Modal llm
    obj_output = multi_modal_llm_program(test_input="hello")
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"
