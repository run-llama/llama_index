from typing import cast

from llama_index.core.readers.loading import load_reader
from llama_index.core.readers.string_iterable import StringIterableReader


def test_loading_readers() -> None:
    string_iterable = StringIterableReader()

    string_iterable_dict = string_iterable.to_dict()

    loaded_string_iterable = cast(
        StringIterableReader, load_reader(string_iterable_dict)
    )

    assert loaded_string_iterable.is_remote == string_iterable.is_remote
