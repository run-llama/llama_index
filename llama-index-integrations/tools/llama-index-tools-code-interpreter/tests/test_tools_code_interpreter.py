from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in CodeInterpreterToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_code_interpreter_documents_host_execution_without_isolation():
    docstring = CodeInterpreterToolSpec.code_interpreter.__doc__

    assert docstring is not None
    assert "subprocess on the machine running this tool" in docstring
    assert "not a sandbox or isolation boundary" in docstring
    assert "executed in isolation" not in docstring
