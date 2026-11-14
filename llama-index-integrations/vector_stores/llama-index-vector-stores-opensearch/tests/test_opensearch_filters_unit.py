from unittest.mock import MagicMock

import pytest

from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.vector_stores.opensearch.base import OpensearchVectorClient


def _make_client() -> OpensearchVectorClient:
    return OpensearchVectorClient(
        endpoint="http://localhost:9200",
        index="test-index",
        dim=4,
        os_client=MagicMock(),
    )


def test_parse_filter_text_match_uses_match_query() -> None:
    client = _make_client()
    flt = MetadataFilter(
        key="name",
        value="John Doe",
        operator=FilterOperator.TEXT_MATCH,
    )

    parsed = client._parse_filter(flt)

    assert parsed == {
        "match": {"metadata.name": {"query": "John Doe", "fuzziness": "AUTO"}}
    }


def test_parse_filter_text_match_insensitive_uses_match_query() -> None:
    client = _make_client()
    flt = MetadataFilter(
        key="name",
        value="john doe",
        operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
    )

    parsed = client._parse_filter(flt)

    assert parsed == {
        "match": {"metadata.name": {"query": "john doe", "fuzziness": "AUTO"}}
    }


TEXT_VALUES = ["Product Management", "Product Marketing"]
NUMERIC_VALUES = [1, 2, 3]


def _terms_set_query(field: str, values: list) -> dict:
    return {
        "terms_set": {
            field: {
                "terms": values,
                "minimum_should_match_script": {"source": "params.num_terms"},
            }
        }
    }


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (FilterOperator.IN, {"terms": {"metadata.category.keyword": TEXT_VALUES}}),
        (FilterOperator.ANY, {"terms": {"metadata.category.keyword": TEXT_VALUES}}),
        (
            FilterOperator.NIN,
            {
                "bool": {
                    "must_not": {"terms": {"metadata.category.keyword": TEXT_VALUES}}
                }
            },
        ),
        (
            FilterOperator.ALL,
            _terms_set_query("metadata.category.keyword", TEXT_VALUES),
        ),
    ],
)
def test_parse_filter_set_operators_use_keyword_for_text_values(
    operator: FilterOperator, expected: dict
) -> None:
    """Every set operator must target the .keyword subfield for text values."""
    client = _make_client()
    flt = MetadataFilter(key="category", value=TEXT_VALUES, operator=operator)

    assert client._parse_filter(flt) == expected


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (FilterOperator.IN, {"terms": {"metadata.score": NUMERIC_VALUES}}),
        (FilterOperator.ANY, {"terms": {"metadata.score": NUMERIC_VALUES}}),
        (
            FilterOperator.NIN,
            {"bool": {"must_not": {"terms": {"metadata.score": NUMERIC_VALUES}}}},
        ),
        (FilterOperator.ALL, _terms_set_query("metadata.score", NUMERIC_VALUES)),
    ],
)
def test_parse_filter_set_operators_skip_keyword_for_non_text_values(
    operator: FilterOperator, expected: dict
) -> None:
    """Numeric values have no .keyword subfield, so the bare field is used."""
    client = _make_client()
    flt = MetadataFilter(key="score", value=NUMERIC_VALUES, operator=operator)

    assert client._parse_filter(flt) == expected
