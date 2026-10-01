from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.python_file import PythonFileToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in PythonFileToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_reads_utf8_source_with_non_ascii_content(tmp_path):
    # Python source defaults to UTF-8 (PEP 3120); comments and string
    # literals are routinely non-ASCII.
    source = "# 中文注释\n\n\ndef greet():\n    return 'héllo ✓'\n"
    path = tmp_path / "sample.py"
    path.write_bytes(source.encode("utf-8"))

    spec = PythonFileToolSpec(str(path))

    assert "greet" in spec.function_definitions()


def test_honors_coding_declaration(tmp_path):
    # PEP 263: a source file may declare a non-UTF-8 encoding.
    source = "# -*- coding: iso-8859-1 -*-\n\n\ndef greet():\n    return 'café'\n"
    path = tmp_path / "legacy.py"
    path.write_bytes(source.encode("iso-8859-1"))

    spec = PythonFileToolSpec(str(path))

    assert "greet" in spec.function_definitions()
