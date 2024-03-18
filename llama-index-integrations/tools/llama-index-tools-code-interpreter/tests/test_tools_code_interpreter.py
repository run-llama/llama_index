from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in CodeInterpreterToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes
