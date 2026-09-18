import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.lilac import LilacReader


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in LilacReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_raises_import_error_when_lilac_missing() -> None:
    with pytest.raises(ImportError, match="lilac"):
        LilacReader().load_data(dataset="namespace/dataset_name")
