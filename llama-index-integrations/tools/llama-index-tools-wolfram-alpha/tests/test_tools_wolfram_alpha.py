from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in WolframAlphaToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes
