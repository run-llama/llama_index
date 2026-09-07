from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.opticparse import OpticParseToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in OpticParseToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_tool_spec_initialization():
    tool = OpticParseToolSpec(api_key="test_key")
    assert tool.api_key == "test_key"
    assert len(tool.spec_functions) == 2
    assert "extract" in tool.spec_functions
    assert "inspect_threat" in tool.spec_functions
