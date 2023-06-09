import pytest
from pydantic import BaseModel

try:
    from guidance.llms import Mock as MockLLM
except ImportError:
    MockLLM = None  # type: ignore
from llama_index.program.guidance_program import GuidancePydanticProgram


@pytest.mark.skipif(MockLLM is None, reason="guidance not installed")
def test_guidance_pydantic_program() -> None:
    class TestModel(BaseModel):
        test_attr: str

    program = GuidancePydanticProgram(
        output_cls=TestModel,
        prompt_template_str="This is a test prompt with a {{test_input}}.",
        llm=MockLLM(),
    )

    assert program.output_cls == TestModel

    output = program(test_input="test_arg")
    assert isinstance(output, TestModel)
