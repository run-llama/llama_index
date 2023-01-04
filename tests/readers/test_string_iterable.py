"""Test String Iterable Reader."""

from gpt_index.readers.string_iterable import StringIterableReader


def test_load() -> None:
    """Test loading data into StringIterableReader."""
    reader = StringIterableReader()
    documents = reader.load_data(texts=["I went to the store", "I bought an apple"])
    assert len(documents) == 2
