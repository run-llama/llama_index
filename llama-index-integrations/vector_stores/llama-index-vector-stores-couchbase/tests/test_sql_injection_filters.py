"""
Regression tests for SQL++ injection in
`_convert_llamaindex_filters_to_sql` (see issue #22314).

These are pure unit tests against a module-level function and require no
live Couchbase connection.
"""

from unittest.mock import MagicMock, patch

from couchbase.cluster import Cluster
from couchbase.options import QueryOptions

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.couchbase.base import (
    CouchbaseQueryVectorStore,
    QueryVectorSearchType,
    QueryVectorSearchSimilarity,
    _convert_llamaindex_filters_to_sql,
)

MALICIOUS_VALUE = "' OR 1=1 UNION SELECT d.* FROM sensitive_bucket d --"


def test_eq_filter_does_not_interpolate_value_into_sql() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value=MALICIOUS_VALUE, operator=FilterOperator.EQ
            )
        ]
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    # The malicious payload must never appear in the generated SQL string.
    assert MALICIOUS_VALUE not in sql
    assert "'" not in sql
    # It must instead be carried as a bound named parameter.
    assert sql == "d.metadata.category = $metadata_filter_0"
    assert params == {"metadata_filter_0": MALICIOUS_VALUE}


def test_ne_gt_lt_filters_use_named_parameters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="status", value="banned", operator=FilterOperator.NE),
            MetadataFilter(key="score", value=10, operator=FilterOperator.GT),
            MetadataFilter(key="score", value=100, operator=FilterOperator.LTE),
        ],
        condition="and",
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    assert sql == (
        "d.metadata.status != $metadata_filter_0 AND "
        "d.metadata.score > $metadata_filter_1 AND "
        "d.metadata.score <= $metadata_filter_2"
    )
    assert params == {
        "metadata_filter_0": "banned",
        "metadata_filter_1": 10,
        "metadata_filter_2": 100,
    }


def test_in_filter_binds_list_as_single_named_parameter() -> None:
    malicious_list_item = "a', 'b') OR 1=1 --"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tag",
                value=["safe", malicious_list_item],
                operator=FilterOperator.IN,
            )
        ]
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    assert malicious_list_item not in sql
    assert sql == "d.metadata.tag IN $metadata_filter_0"
    assert params == {"metadata_filter_0": ["safe", malicious_list_item]}


def test_in_filter_requires_list_value() -> None:
    import pytest

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tag", value="not-a-list", operator=FilterOperator.IN)
        ]
    )

    with pytest.raises(ValueError, match="expects a list value"):
        _convert_llamaindex_filters_to_sql(filters, "metadata")


def test_nested_filters_produce_unique_parameter_names() -> None:
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=MALICIOUS_VALUE, operator=FilterOperator.EQ),
            MetadataFilter(key="b", value="x", operator=FilterOperator.EQ),
        ],
        condition="or",
    )
    outer = MetadataFilters(
        filters=[
            MetadataFilter(key="c", value=1, operator=FilterOperator.GTE),
            inner,
        ],
        condition="and",
    )

    sql, params = _convert_llamaindex_filters_to_sql(outer, "metadata")

    # Three total leaf filters -> three unique parameter names, no reuse.
    assert len(params) == 3
    assert MALICIOUS_VALUE not in sql
    assert sql == (
        "d.metadata.c >= $metadata_filter_0 AND "
        "(d.metadata.a = $metadata_filter_1 OR d.metadata.b = $metadata_filter_2)"
    )


def test_empty_filters_return_empty_sql_and_params() -> None:
    sql, params = _convert_llamaindex_filters_to_sql(None, "metadata")
    assert sql == ""
    assert params == {}


# ---------------------------------------------------------------------------
# query()-level tests: exercise the full path, including the named_parameters
# merge in CouchbaseQueryVectorStore.query(), which the unit tests above
# (against _convert_llamaindex_filters_to_sql directly) don't cover. These use
# a mocked Couchbase cluster - no live connection needed.
# ---------------------------------------------------------------------------


def _make_mock_cluster() -> MagicMock:
    """
    A Mock cluster that satisfies isinstance(cluster, Cluster) (via spec) and
    returns an empty result set from .query(), so CouchbaseQueryVectorStore
    construction and query() execution succeed without a real connection.
    """
    cluster = MagicMock(spec=Cluster)
    mock_result = MagicMock()
    mock_result.rows.return_value = []
    cluster.query.return_value = mock_result
    return cluster


def _make_vector_store(cluster: MagicMock, **kwargs) -> CouchbaseQueryVectorStore:
    """
    Construct a CouchbaseQueryVectorStore against a mocked cluster.

    The bucket/scope/collection-existence checks in __init__ walk the real
    Couchbase SDK's bucket.collections().get_all_scopes() shape, which isn't
    meaningful to fake for a mocked cluster and isn't what these tests are
    verifying - so they're patched out directly, isolating the tests to the
    filter/query-building logic under test.
    """
    with (
        patch.object(
            CouchbaseQueryVectorStore, "_check_bucket_exists", return_value=True
        ),
        patch.object(
            CouchbaseQueryVectorStore,
            "_check_scope_and_collection_exists",
            return_value=True,
        ),
    ):
        return CouchbaseQueryVectorStore(
            cluster=cluster,
            bucket_name="test_bucket",
            scope_name="test_scope",
            collection_name="test_collection",
            search_type=QueryVectorSearchType.ANN,
            similarity=QueryVectorSearchSimilarity.COSINE,
            **kwargs,
        )


def test_query_with_malicious_filter_does_not_interpolate_into_sql() -> None:
    """
    End-to-end regression test for #22314: the malicious payload from the
    issue must never appear in the SQL string actually sent to the cluster,
    and must instead be bound as a named parameter.
    """
    cluster = _make_mock_cluster()
    store = _make_vector_store(cluster)

    q = VectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value=MALICIOUS_VALUE, operator=FilterOperator.EQ
                )
            ]
        ),
    )

    store.query(q)

    assert cluster.query.call_count == 1
    query_str, query_options_arg = cluster.query.call_args.args

    assert MALICIOUS_VALUE not in query_str
    assert query_options_arg["named_parameters"] == {
        "metadata_filter_0": MALICIOUS_VALUE
    }


def test_query_merges_filter_params_without_clobbering_existing_named_parameters() -> (
    None
):
    """
    Verifies the claim in the fix: pre-existing named_parameters set on
    self._query_options must survive being merged with filter-derived ones,
    not be overwritten by them.
    """
    cluster = _make_mock_cluster()
    store = _make_vector_store(
        cluster,
        query_options=QueryOptions(
            named_parameters={"existing_param": "existing_value"}
        ),
    )

    q = VectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        similarity_top_k=1,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="genre", value="Thriller", operator=FilterOperator.EQ
                )
            ]
        ),
    )

    store.query(q)

    _, query_options_arg = cluster.query.call_args.args
    merged = query_options_arg["named_parameters"]
    assert merged["existing_param"] == "existing_value"
    assert merged["metadata_filter_0"] == "Thriller"


def test_query_without_filters_does_not_set_named_parameters() -> None:
    """
    A query with no filters shouldn't fabricate an empty named_parameters
    entry or otherwise disturb query_options.
    """
    cluster = _make_mock_cluster()
    store = _make_vector_store(cluster)

    q = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=1)
    store.query(q)

    _, query_options_arg = cluster.query.call_args.args
    assert not query_options_arg.get("named_parameters")
