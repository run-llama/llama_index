import pytest
from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.output_parsers.base import OutputParserException

try:
    from guidance.models import Mock as MockLLM
except ImportError:
    MockLLM = None  # type: ignore
from llama_index.legacy.program.guidance_program import GuidancePydanticProgram


@pytest.mark.skipif(MockLLM is None, reason="guidance not installed")
def test_guidance_pydantic_program() -> None:
    class TestModel(BaseModel):
        test_attr: str

    program = GuidancePydanticProgram(
        output_cls=TestModel,
        prompt_template_str="This is a test prompt with a {{test_input}}.",
        guidance_llm=MockLLM(),
    )

    assert program.output_cls == TestModel

    with pytest.raises(OutputParserException):
        _ = program(tools_str="test_tools", query_str="test_query")
