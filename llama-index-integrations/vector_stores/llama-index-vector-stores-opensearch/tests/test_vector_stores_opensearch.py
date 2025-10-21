from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from typing import Optional
import pytest
from unittest.mock import MagicMock, patch


def test_class():
    names_of_base_classes = [b.__name__ for b in OpensearchVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    ("is_aoss", "mock_version", "expected_result"),
    [
        (True, None, False),
        (False, "3.0.0", True),
        (False, "2.9.0", True),
        (False, "2.11.1", True),
        (False, "2.8.0", False),
        (False, "1.3.7", False),
    ],
)
@patch("llama_index.vector_stores.opensearch.OpensearchVectorClient.__init__")
def test_is_efficient_filtering_enabled_version_logic(
    mock_init: MagicMock,
    is_aoss: bool,
    mock_version: Optional[str],
    expected_result: bool,
):
    """
    Verifies the version checking logic within _is_efficient_filtering_enabled.
    This test covers the line change:
    `ef_enabled = int(major) > 2 or (int(major) == 2 and int(minor) >= 9)`
    """
    mock_init.return_value = None
    client = OpensearchVectorClient()
    mock_init.assert_called_once()

    client.is_aoss = is_aoss

    with patch.object(
        client, "_get_opensearch_version", return_value=mock_version
    ) as mock_get_version:
        result = client._is_efficient_filtering_enabled()

        assert result == expected_result

        if is_aoss:
            mock_get_version.assert_not_called()
        else:
            mock_get_version.assert_called_once()


@pytest.mark.parametrize(
    ("efficient_filtering_enabled", "expected_path"),
    [
        (True, "approximate_search"),
        (False, "script_score"),
    ],
)
@patch("llama_index.vector_stores.opensearch.OpensearchVectorClient.__init__")
def test_knn_search_query_routing_with_filters(
    mock_init: MagicMock, efficient_filtering_enabled: bool, expected_path: str
):
    """
    Verifies that _knn_search_query correctly routes to the appropriate
    internal query method based on the _efficient_filtering_enabled flag.

    This test works by mocking the real OpensearchVectorClient's __init__ to
    prevent any network calls. It then manually sets the internal state
    (_efficient_filtering_enabled) to control which code path is taken in
    the method under test.
    """
    mock_init.return_value = None

    client = OpensearchVectorClient()

    client.is_aoss = False
    client._efficient_filtering_enabled = efficient_filtering_enabled
    client._method = {"engine": "lucene"}

    client._default_approximate_search_query = MagicMock()
    client._default_scoring_script_query = MagicMock()
    client._parse_filters = MagicMock(return_value={"term": {"field": "value"}})

    client._knn_search_query(
        embedding_field="test_embedding",
        query_embedding=[1.0, 2.0, 3.0],
        k=10,
        filters=MetadataFilters(filters=[MetadataFilter(key="foo", value="bar")]),
    )

    expected_filter_structure = {"bool": {"filter": client._parse_filters.return_value}}
    mock_init.assert_called_once()

    if expected_path == "approximate_search":
        client._default_approximate_search_query.assert_called_once()
        client._default_scoring_script_query.assert_not_called()

        _, kwargs = client._default_approximate_search_query.call_args
        assert kwargs.get("filters") == expected_filter_structure
        assert "pre_filter" not in kwargs

    elif expected_path == "script_score":
        client._default_scoring_script_query.assert_called_once()
        _, kwargs = client._default_scoring_script_query.call_args
        assert kwargs.get("pre_filter") == expected_filter_structure
        assert "filters" not in kwargs
