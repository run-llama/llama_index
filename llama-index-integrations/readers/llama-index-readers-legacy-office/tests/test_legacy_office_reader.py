from pathlib import Path
from types import SimpleNamespace

import pytest

from llama_index.readers.legacy_office import LegacyOfficeReader


def _reader_with_parsed(parsed):
    # Skip __init__ so no Tika server is needed; only load_data is exercised.
    reader = LegacyOfficeReader.__new__(LegacyOfficeReader)
    reader.parser = SimpleNamespace(from_file=lambda _path: parsed)
    reader.excluded_embed_metadata_keys = []
    reader.excluded_llm_metadata_keys = []
    return reader


def test_load_data_returns_content():
    reader = _reader_with_parsed({"content": "\n hello \n", "metadata": {}})

    docs = reader.load_data(Path("a.doc"))

    assert [d.text for d in docs] == ["hello"]


@pytest.mark.parametrize("content", [None, "", "  \n "])
def test_load_data_empty_content_raises(content):
    # Tika reports content=None for a document without text.
    reader = _reader_with_parsed({"content": content, "metadata": {}})

    with pytest.raises(ValueError, match="No content found"):
        reader.load_data(Path("empty.doc"))
