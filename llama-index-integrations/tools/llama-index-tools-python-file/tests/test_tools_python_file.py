import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.python_file import PythonFileToolSpec

SAMPLE_SOURCE = '''
def sync_fn(a, b=1):
    """Doc for sync."""
    return a


async def async_fn(x):
    """Doc for async."""
    return x


def _private_fn():
    pass
'''


@pytest.fixture()
def python_file(tmp_path):
    path = tmp_path / "sample_module.py"
    path.write_text(SAMPLE_SOURCE, encoding="utf-8")
    return path


def test_class():
    names_of_base_classes = [b.__name__ for b in PythonFileToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_function_definitions_include_async_functions(python_file):
    definitions = PythonFileToolSpec(str(python_file)).function_definitions()

    assert "sync_fn" in definitions
    assert "async_fn" in definitions


def test_get_function_finds_async_functions(python_file):
    definition = PythonFileToolSpec(str(python_file)).get_function("async_fn")

    assert definition is not None
    assert "async_fn" in definition
    assert "Doc for async." in definition


def test_get_functions_skips_unknown_names(python_file):
    definitions = PythonFileToolSpec(str(python_file)).get_functions(
        ["sync_fn", "does_not_exist"]
    )

    assert "sync_fn" in definitions
    assert "does_not_exist" not in definitions
