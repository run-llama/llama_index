from llama_index.core.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.program.lmformatenforcer import LMFormatEnforcerPydanticProgram


def test_class():
    names_of_base_classes = [
        b.__name__ for b in LMFormatEnforcerPydanticProgram.__mro__
    ]
    assert BaseLLMFunctionProgram.__name__ in names_of_base_classes
