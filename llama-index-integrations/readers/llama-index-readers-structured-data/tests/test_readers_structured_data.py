from pathlib import Path

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.structured_data.base import StructuredDataReader


def test_class():
    names_of_base_classes = [b.__name__ for b in StructuredDataReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@pytest.fixture()
def people_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "people.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,40\n", encoding="utf-8")
    return csv_file


def test_first_column_as_text(people_csv: Path) -> None:
    """Column index 0 is a valid selector, not a missing col_index."""
    documents = StructuredDataReader(col_index=0).load_data(people_csv)

    assert [document.text for document in documents] == ["Alice", "Bob"]


def test_first_column_as_metadata(people_csv: Path) -> None:
    """Column index 0 in col_metadata must reach the document metadata."""
    documents = StructuredDataReader(col_index="name", col_metadata=0).load_data(
        people_csv
    )

    assert [document.text for document in documents] == ["Alice", "Bob"]
    assert [document.metadata for document in documents] == [
        {"name": "Alice"},
        {"name": "Bob"},
    ]


def test_missing_col_index_still_raises(people_csv: Path) -> None:
    with pytest.raises(ValueError, match="col_index must be specified"):
        StructuredDataReader(col_index=None).load_data(people_csv)
