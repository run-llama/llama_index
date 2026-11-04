from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.json import JSONReader


def test_class():
    names_of_base_classes = [b.__name__ for b in JSONReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@pytest.mark.parametrize("metadata", [None, {"source": "local"}])
@pytest.mark.parametrize(
    ("options", "expected_text"),
    [
        ({}, '"name": "Alice"'),
        ({"clean_json": False}, '{"name": "Alice"}'),
        ({"levels_back": 0}, "name Alice"),
    ],
)
def test_load_data_accepts_optional_metadata(
    tmp_path: Path,
    metadata: Optional[Dict[str, str]],
    options: Dict[str, Any],
    expected_text: str,
) -> None:
    input_file = tmp_path / "sample.json"
    input_file.write_text('{"name": "Alice"}', encoding="utf-8")

    documents = JSONReader(**options).load_data(str(input_file), extra_info=metadata)

    assert len(documents) == 1
    assert documents[0].text == expected_text
    assert documents[0].metadata == (metadata or {})
