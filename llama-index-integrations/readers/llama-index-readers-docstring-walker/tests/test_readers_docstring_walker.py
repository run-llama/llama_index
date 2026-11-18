from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.readers.docstring_walker import DocstringWalker

MODULE_SOURCE = '''
"""Module docstring."""


def sync_function():
    """Docstring of sync_function."""


async def async_function():
    """Docstring of async_function."""


class SomeClass:
    """Docstring of SomeClass."""

    def sync_method(self):
        """Docstring of sync_method."""

    async def async_method(self):
        """Docstring of async_method."""
'''


def test_class():
    names_of_base_classes = [b.__name__ for b in DocstringWalker.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _walk(tmp_path: Path) -> str:
    (tmp_path / "sample.py").write_text(MODULE_SOURCE, encoding="utf-8")
    documents = DocstringWalker().load_data(str(tmp_path))
    assert len(documents) == 1
    return documents[0].text


def test_extracts_sync_docstrings(tmp_path: Path) -> None:
    text = _walk(tmp_path)

    assert "Docstring of sync_function." in text
    assert "Docstring of sync_method." in text


def test_extracts_async_function_docstrings(tmp_path: Path) -> None:
    text = _walk(tmp_path)

    assert "Docstring of async_function." in text


def test_extracts_async_method_docstrings(tmp_path: Path) -> None:
    text = _walk(tmp_path)

    assert "Docstring of async_method." in text
