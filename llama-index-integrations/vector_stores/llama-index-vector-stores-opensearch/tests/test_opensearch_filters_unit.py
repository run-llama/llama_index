import json
from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
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


def test_parse_filter_key_prefixes_and_keyword_suffixes() -> None:
    client = _make_client()

    # Plain key with string value
    flt1 = MetadataFilter(key="file_name", value="sample.pdf")
    assert client._parse_filter(flt1) == {
        "term": {"metadata.file_name.keyword": "sample.pdf"}
    }

    # Key already starting with metadata.
    flt2 = MetadataFilter(key="metadata.file_name", value="sample.pdf")
    assert client._parse_filter(flt2) == {
        "term": {"metadata.file_name.keyword": "sample.pdf"}
    }

    # Key already ending with .keyword
    flt3 = MetadataFilter(key="file_name.keyword", value="sample.pdf")
    assert client._parse_filter(flt3) == {
        "term": {"metadata.file_name.keyword": "sample.pdf"}
    }

    # Key starting with metadata. and ending with .keyword
    flt4 = MetadataFilter(key="metadata.file_name.keyword", value="sample.pdf")
    assert client._parse_filter(flt4) == {
        "term": {"metadata.file_name.keyword": "sample.pdf"}
    }


def test_parse_filter_numeric_types() -> None:
    client = _make_client()

    # Integer value -> no .keyword
    flt1 = MetadataFilter(key="page_label", value=2)
    assert client._parse_filter(flt1) == {"term": {"metadata.page_label": 2}}

    # Float value in range operator
    flt3 = MetadataFilter(key="score", value=0.75, operator=FilterOperator.GTE)
    assert client._parse_filter(flt3) == {"range": {"metadata.score": {"gte": 0.75}}}


def test_parse_filter_json_encoded_values() -> None:
    client = _make_client()

    # JSON-encoded string
    flt1 = MetadataFilter(key="file_name", value=json.dumps("sample.pdf"))
    assert client._parse_filter(flt1) == {
        "term": {"metadata.file_name.keyword": "sample.pdf"}
    }

    # JSON-encoded string list for IN
    flt2 = MetadataFilter(
        key="tags",
        value=json.dumps(["tag1", "tag2"]),
        operator=FilterOperator.IN,
    )
    assert client._parse_filter(flt2) == {
        "terms": {"metadata.tags.keyword": ["tag1", "tag2"]}
    }


def test_parse_filter_operators() -> None:
    client = _make_client()

    # NE operator
    flt_ne = MetadataFilter(key="status", value="archived", operator=FilterOperator.NE)
    assert client._parse_filter(flt_ne) == {
        "bool": {"must_not": {"term": {"metadata.status.keyword": "archived"}}}
    }

    # IN operator with string list
    flt_in = MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.IN)
    assert client._parse_filter(flt_in) == {
        "terms": {"metadata.tags.keyword": ["a", "b"]}
    }

    # IN operator with int list
    flt_in_int = MetadataFilter(key="ids", value=[1, 2], operator=FilterOperator.IN)
    assert client._parse_filter(flt_in_int) == {"terms": {"metadata.ids": [1, 2]}}

    # NIN operator with string list
    flt_nin = MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.NIN)
    assert client._parse_filter(flt_nin) == {
        "bool": {"must_not": {"terms": {"metadata.tags.keyword": ["a", "b"]}}}
    }

    # ALL operator with string list
    flt_all = MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.ALL)
    assert client._parse_filter(flt_all) == {
        "terms_set": {
            "metadata.tags.keyword": {
                "terms": ["a", "b"],
                "minimum_should_match_script": {"source": "params.num_terms"},
            }
        }
    }

    # CONTAINS operator
    flt_contains = MetadataFilter(
        key="summary", value="test", operator=FilterOperator.CONTAINS
    )
    assert client._parse_filter(flt_contains) == {
        "wildcard": {"metadata.summary.keyword": "*test*"}
    }

    # IS_EMPTY operator
    flt_empty = MetadataFilter(
        key="notes", value=None, operator=FilterOperator.IS_EMPTY
    )
    assert client._parse_filter(flt_empty) == {
        "bool": {"must_not": {"exists": {"field": "metadata.notes"}}}
    }


def test_parse_filters_recursively_nested() -> None:
    client = _make_client()

    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="category", value="finance"),
            MetadataFilters(
                condition=FilterCondition.OR,
                filters=[
                    MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
                    MetadataFilter(key="priority", value="high"),
                ],
            ),
        ],
    )

    parsed = client._parse_filters_recursively(filters)

    assert parsed == {
        "bool": {
            "must": [
                {"term": {"metadata.category.keyword": "finance"}},
                {
                    "bool": {
                        "should": [
                            {"range": {"metadata.year": {"gte": 2024}}},
                            {"term": {"metadata.priority.keyword": "high"}},
                        ]
                    }
                },
            ]
        }
    }


def test_coerce_filter_value() -> None:
    # Plain strings without outer JSON quotes should remain unchanged
    assert OpensearchVectorClient._coerce_filter_value("sample.pdf") == "sample.pdf"
    assert OpensearchVectorClient._coerce_filter_value("hello world") == "hello world"

    # JSON strings
    assert OpensearchVectorClient._coerce_filter_value('"sample.pdf"') == "sample.pdf"
    assert OpensearchVectorClient._coerce_filter_value('["a", "b"]') == ["a", "b"]
    assert OpensearchVectorClient._coerce_filter_value("true") is True
    assert OpensearchVectorClient._coerce_filter_value("false") is False
    assert OpensearchVectorClient._coerce_filter_value("null") is None

    # Native types
    assert OpensearchVectorClient._coerce_filter_value(123) == 123
    assert OpensearchVectorClient._coerce_filter_value(True) is True
    assert OpensearchVectorClient._coerce_filter_value(["a", "b"]) == ["a", "b"]
