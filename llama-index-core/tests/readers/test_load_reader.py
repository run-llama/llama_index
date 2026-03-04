from typing import cast

import pytest
from llama_index.core.readers.loading import load_reader
from llama_index.core.readers.string_iterable import StringIterableReader


def test_loading_readers() -> None:
    string_iterable = StringIterableReader()
    assert load_reader(string_iterable) == string_iterable  # type: ignore

    string_iterable_dict = string_iterable.to_dict()

    loaded_string_iterable = cast(
        StringIterableReader, load_reader(string_iterable_dict)
    )

    assert loaded_string_iterable.is_remote == string_iterable.is_remote

    string_iterable_dict.pop("class_name")
    with pytest.raises(ValueError, match="Must specify `class_name` in reader data."):
        load_reader(string_iterable_dict)

    string_iterable_dict["class_name"] = "doesnt_exist"
    with pytest.raises(ValueError, match="Reader class name doesnt_exist not found."):
        load_reader(string_iterable_dict)
