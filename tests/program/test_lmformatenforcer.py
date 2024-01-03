from importlib.util import find_spec
from unittest.mock import MagicMock

import pytest
from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.types import CompletionResponse
from llama_index.program.lmformatenforcer_program import LMFormatEnforcerPydanticProgram

has_lmformatenforcer = find_spec("lmformatenforcer") is not None


@pytest.mark.skipif(not has_lmformatenforcer, reason="lm-format-enforcer not installed")
def test_lmformatenforcer_pydantic_program() -> None:
    class TestModel(BaseModel):
        test_attr: str

    prompt = "This is a test prompt with a {test_input}."
    generated_text = '{"test_attr": "blue"}'
    test_value = "test_arg"

    llm = MagicMock(spec=HuggingFaceLLM)
    llm.complete.return_value = CompletionResponse(text=generated_text)
    llm.generate_kwargs = {}

    program = LMFormatEnforcerPydanticProgram(
        output_cls=TestModel, prompt_template_str=prompt, llm=llm
    )

    output = program(test_input=test_value)
    assert isinstance(output, TestModel)
    assert output.test_attr == "blue"
