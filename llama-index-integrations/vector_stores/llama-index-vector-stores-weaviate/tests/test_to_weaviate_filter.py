from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.vector_stores.weaviate.base import (
    _coerce_filter_value,
    _to_weaviate_filter,
)
from weaviate.classes.config import DataType


def test_to_weaviate_filter_with_various_operators():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1)])
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "Equal"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.NE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "NotEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GT)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "GreaterThan"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GTE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "GreaterThanEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LT)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "LessThan"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LTE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "LessThanEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "IsNull"
    assert filter.value is True


def test_to_weaviate_filter_with_multiple_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.GTE),
            MetadataFilter(key="a", value=10, operator=FilterOperator.LTE),
        ],
        condition=FilterCondition.AND,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "And"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "GreaterThanEqual"
    assert filter.filters[0].value == 1
    assert filter.filters[1].target == "a"
    assert filter.filters[1].operator == "LessThanEqual"
    assert filter.filters[1].value == 10

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.LT),
            MetadataFilter(key="a", value=10, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.OR,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "Or"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "LessThan"
    assert filter.filters[0].value == 1
    assert filter.filters[1].target == "a"
    assert filter.filters[1].operator == "GreaterThan"
    assert filter.filters[1].value == 10


def test_to_weaviate_filter_with_nested_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
                    MetadataFilter(key="c", value=3, operator=FilterOperator.GT),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "And"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "Equal"
    assert filter.filters[0].value == 1
    assert filter.filters[1].operator == "Or"
    or_filters = filter.filters[1].filters
    assert len(or_filters) == 2
    assert or_filters[0].target == "b"
    assert or_filters[0].operator == "Equal"
    assert or_filters[0].value == 2
    assert or_filters[1].target == "c"
    assert or_filters[1].operator == "GreaterThan"
    assert or_filters[1].value == 3


class TestCoerceFilterValue:
    """Tests for _coerce_filter_value."""

    def test_int_to_float_for_number_type(self):
        """An int value should be coerced to float for a 'number' property."""
        result = _coerce_filter_value(5, "number")
        assert result == 5.0
        assert isinstance(result, float)

    def test_float_to_int_for_int_type(self):
        """A float value should be coerced to int for an 'int' property."""
        result = _coerce_filter_value(5.0, "int")
        assert result == 5
        assert isinstance(result, int)

    def test_numeric_string_stays_string_for_text_type(self):
        """A numeric string should stay as string for a 'text' property."""
        result = _coerce_filter_value("1234", "text")
        assert result == "1234"
        assert isinstance(result, str)

    def test_int_to_string_for_text_type(self):
        """An int value should be coerced to string for a 'text' property."""
        result = _coerce_filter_value(42, "text")
        assert result == "42"
        assert isinstance(result, str)

    def test_string_to_float_for_number_type(self):
        """A numeric string should be coerced to float for a 'number' property."""
        result = _coerce_filter_value("3.14", "number")
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_to_int_for_int_type(self):
        """A numeric string should be coerced to int for an 'int' property."""
        result = _coerce_filter_value("42", "int")
        assert result == 42
        assert isinstance(result, int)

    def test_non_numeric_string_for_number_type_returns_original(self):
        """A non-numeric string should be returned as-is when coercion fails."""
        result = _coerce_filter_value("hello", "number")
        assert result == "hello"
        assert isinstance(result, str)

    def test_list_coercion_to_float(self):
        """A list of ints should be coerced to floats for 'number[]' property."""
        result = _coerce_filter_value([1, 2, 3], "number[]")
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    def test_list_coercion_to_string(self):
        """A list of ints should be coerced to strings for 'text[]' property."""
        result = _coerce_filter_value([1, 2, 3], "text[]")
        assert result == ["1", "2", "3"]
        assert all(isinstance(v, str) for v in result)

    def test_unknown_data_type_returns_original(self):
        """An unknown data type should return the value unchanged."""
        result = _coerce_filter_value(42, "geoCoordinates")
        assert result == 42
        assert isinstance(result, int)

    def test_none_value_returns_none(self):
        """None value should pass through unchanged."""
        result = _coerce_filter_value(None, "text")
        assert result is None

    def test_bool_coercion(self):
        """Integer should be coerced to boolean for 'boolean' property."""
        result = _coerce_filter_value(1, "boolean")
        assert result is True
        assert isinstance(result, bool)

    def test_string_false_to_bool(self):
        """String 'false' should be coerced to False, not True."""
        result = _coerce_filter_value("false", "boolean")
        assert result is False
        assert isinstance(result, bool)

    def test_string_true_to_bool(self):
        """String 'true' should be coerced to True."""
        result = _coerce_filter_value("true", "boolean")
        assert result is True
        assert isinstance(result, bool)

    def test_string_zero_to_bool(self):
        """String '0' should be coerced to False."""
        result = _coerce_filter_value("0", "boolean")
        assert result is False
        assert isinstance(result, bool)

    def test_unrecognised_string_for_bool_returns_original(self):
        """An unrecognised string should be returned as-is for 'boolean'."""
        result = _coerce_filter_value("maybe", "boolean")
        assert result == "maybe"
        assert isinstance(result, str)

    def test_bool_list_with_strings(self):
        """A list of string booleans should be parsed correctly for 'boolean[]'."""
        result = _coerce_filter_value(["true", "false", "1", "0"], "boolean[]")
        assert result == [True, False, True, False]
        assert all(isinstance(v, bool) for v in result)

    def test_with_datatype_number_enum(self):
        """DataType.NUMBER enum should be handled the same as string 'number'."""
        result = _coerce_filter_value(5, DataType.NUMBER)
        assert result == 5.0
        assert isinstance(result, float)

    def test_with_datatype_text_enum(self):
        """DataType.TEXT enum should be handled the same as string 'text'."""
        result = _coerce_filter_value(42, DataType.TEXT)
        assert result == "42"
        assert isinstance(result, str)

    def test_with_datatype_int_enum(self):
        """DataType.INT enum should be handled the same as string 'int'."""
        result = _coerce_filter_value(5.0, DataType.INT)
        assert result == 5
        assert isinstance(result, int)

    def test_with_datatype_bool_enum(self):
        """DataType.BOOL enum should be handled the same as string 'boolean'."""
        result = _coerce_filter_value(1, DataType.BOOL)
        assert result is True
        assert isinstance(result, bool)

    def test_with_datatype_number_array_enum(self):
        """DataType.NUMBER_ARRAY enum should coerce list of ints to floats."""
        result = _coerce_filter_value([1, 2, 3], DataType.NUMBER_ARRAY)
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)


class TestToWeaviateFilterWithPropertyTypes:
    """Tests for _to_weaviate_filter with property_types parameter."""

    def test_int_coerced_to_float_for_number_property(self):
        """An int filter value should be coerced to float when the schema property is 'number'."""
        filters = MetadataFilters(filters=[MetadataFilter(key="score", value=5)])
        property_types = {"score": "number"}
        result = _to_weaviate_filter(filters, property_types)
        assert result.target == "score"
        assert result.value == 5.0
        assert isinstance(result.value, float)

    def test_numeric_string_stays_string_for_text_property(self):
        """A numeric string filter value should stay as string when the schema property is 'text'."""
        filters = MetadataFilters(
            filters=[MetadataFilter(key="article_id", value="1234")]
        )
        property_types = {"article_id": "text"}
        result = _to_weaviate_filter(filters, property_types)
        assert result.target == "article_id"
        assert result.value == "1234"
        assert isinstance(result.value, str)

    def test_no_property_types_preserves_original_value(self):
        """Without property_types, values should pass through unchanged."""
        filters = MetadataFilters(filters=[MetadataFilter(key="score", value=5)])
        result = _to_weaviate_filter(filters)
        assert result.value == 5
        assert isinstance(result.value, int)

    def test_property_not_in_types_preserves_original_value(self):
        """A filter key not found in property_types should preserve its value."""
        filters = MetadataFilters(
            filters=[MetadataFilter(key="unknown_field", value=5)]
        )
        property_types = {"other_field": "text"}
        result = _to_weaviate_filter(filters, property_types)
        assert result.value == 5
        assert isinstance(result.value, int)

    def test_is_empty_not_coerced(self):
        """IS_EMPTY operator should set value to True regardless of property_types."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value=None, operator=FilterOperator.IS_EMPTY)
            ]
        )
        property_types = {"a": "text"}
        result = _to_weaviate_filter(filters, property_types)
        assert result.value is True

    def test_multiple_filters_with_mixed_types(self):
        """Multiple filters with different types should each be coerced correctly."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="score", value=5, operator=FilterOperator.GTE),
                MetadataFilter(key="name", value=42, operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        property_types = {"score": "number", "name": "text"}
        result = _to_weaviate_filter(filters, property_types)
        assert result.operator == "And"
        assert result.filters[0].value == 5.0
        assert isinstance(result.filters[0].value, float)
        assert result.filters[1].value == "42"
        assert isinstance(result.filters[1].value, str)

    def test_nested_filters_with_property_types(self):
        """Property types should be threaded through nested MetadataFilters."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="score", value=5, operator=FilterOperator.EQ),
                MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="level", value=3, operator=FilterOperator.GT
                        ),
                    ],
                    condition=FilterCondition.OR,
                ),
            ],
            condition=FilterCondition.AND,
        )
        property_types = {"score": "number", "level": "number"}
        result = _to_weaviate_filter(filters, property_types)
        assert isinstance(result.filters[0].value, float)
        assert result.filters[0].value == 5.0
        nested = result.filters[1]
        assert isinstance(nested.value, float)
        assert nested.value == 3.0
