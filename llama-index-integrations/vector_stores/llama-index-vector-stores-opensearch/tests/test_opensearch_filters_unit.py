from unittest.mock import MagicMock

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
