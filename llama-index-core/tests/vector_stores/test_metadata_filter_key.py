import pytest
from pydantic import ValidationError

from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters


@pytest.mark.parametrize("key", ["", "   ", "\t"])
def test_a_filter_without_a_key_is_rejected(key):
    with pytest.raises(ValidationError, match="must name the metadata field"):
        MetadataFilter(key=key, value="x")


def test_from_dict_without_a_key_is_rejected():
    with pytest.raises(ValidationError, match="must name the metadata field"):
        MetadataFilter.from_dict({"key": " ", "value": "x"})


def test_a_filter_list_without_a_key_is_rejected():
    with pytest.raises(ValidationError, match="must name the metadata field"):
        MetadataFilters(filters=[MetadataFilter(key="", value="x")])


def test_a_filter_with_a_key_still_works():
    metadata_filter = MetadataFilter(key="category", value="books")

    assert metadata_filter.key == "category"
    assert MetadataFilters(filters=[metadata_filter]).filters[0].key == "category"
