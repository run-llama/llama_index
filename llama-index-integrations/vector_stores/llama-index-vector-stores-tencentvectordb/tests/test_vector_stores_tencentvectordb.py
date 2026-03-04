from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.tencentvectordb import (
    TencentVectorDB,
    CollectionParams,
    FilterField,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in TencentVectorDB.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_collection_params_filter_fields():
    """
    Test that CollectionParams.filter_fields property works correctly.

    This test verifies the fix for Issue #19675 where filter_fields was not accessible.
    """
    # Test with filter fields
    filter_field = FilterField(name="author", data_type="string")
    collection_params = CollectionParams(dimension=1536, filter_fields=[filter_field])

    # Test that filter_fields property is accessible
    assert hasattr(collection_params, "filter_fields")
    assert len(collection_params.filter_fields) == 1
    assert collection_params.filter_fields[0].name == "author"
    assert collection_params.filter_fields[0].data_type == "string"

    # Test iteration over filter_fields (this was failing before the fix)
    for field in collection_params.filter_fields:
        assert field.name == "author"
        assert field.data_type == "string"

    # Test with empty filter fields
    empty_params = CollectionParams(dimension=1536)
    assert len(empty_params.filter_fields) == 0

    # Test that empty filter_fields can be iterated over
    for field in empty_params.filter_fields:
        # This should not raise an AttributeError
        pass
