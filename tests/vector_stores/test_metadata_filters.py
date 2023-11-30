import pytest
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


def test_legacy_filters_value_error() -> None:
    """Test legacy filters."""
    filters = [
        MetadataFilter(key="key1", value="value1", operator=FilterOperator.GTE),
        MetadataFilter(key="key2", value="value2"),
        ExactMatchFilter(key="key3", value="value3"),
    ]
    metadata_filters = MetadataFilters(filters=filters)

    with pytest.raises(ValueError):
        metadata_filters.legacy_filters()


def test_legacy_filters() -> None:
    filters = [
        ExactMatchFilter(key="key1", value="value1"),
        ExactMatchFilter(key="key2", value="value2"),
    ]
    metadata_filters = MetadataFilters(filters=filters)
    legacy_filters = metadata_filters.legacy_filters()

    assert len(legacy_filters) == 2
    assert legacy_filters[0].key == "key1"
    assert legacy_filters[0].value == "value1"
    assert legacy_filters[1].key == "key2"
    assert legacy_filters[1].value == "value2"
