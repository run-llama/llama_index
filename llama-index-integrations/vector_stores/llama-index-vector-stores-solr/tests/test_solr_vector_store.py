"""Test OSS Apache Solr vector store."""

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.solr.base import (
    ApacheSolrVectorStore,
)
from llama_index.vector_stores.solr.client.async_ import AsyncSolrClient
from llama_index.vector_stores.solr.client.responses import (
    SolrResponseHeader,
    SolrSelectResponse,
    SolrSelectResponseBody,
)
from llama_index.vector_stores.solr.client.sync import SyncSolrClient
from llama_index.vector_stores.solr.types import BoostedTextField, SolrQueryDict

# Test parameter decorators for reuse
params_add_kwargs = pytest.mark.parametrize(
    "add_kwargs",
    [{"some": "arg"}, {}],
    ids=["Has add_kwargs", "No add_kwargs"],
)

params_delete_kwargs = pytest.mark.parametrize(
    "delete_kwargs",
    [{"some": "arg"}, {}],
    ids=["Has delete_kwargs", "No delete_kwargs"],
)


@pytest.fixture
def mock_sync_client() -> MagicMock:
    """Mock synchronous Solr client."""
    return MagicMock(spec=SyncSolrClient)


@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Mock asynchronous Solr client."""
    return AsyncMock(spec=AsyncSolrClient)


@pytest.fixture
def mock_solr_response_docs() -> list[dict[str, Any]]:
    """Mock Solr response documents."""
    return [
        {
            "id": "node0",
            "contents": "some text",
            "embedding": [0.1, 0.2, 0.3],
            "extra_field": "extra field",
            "other_extra_field": "other extra field",
            "score": 0.95,
        },
        {
            "id": "node1",
            "contents": "some text",
            "embedding": [0.1, 0.2, 0.3],
            "extra_field": "extra field",
            "other_extra_field": "other extra field",
            "score": 0.85,
        },
    ]


@pytest.fixture
def mock_solr_response(
    mock_solr_response_docs: list[dict[str, Any]],
) -> SolrSelectResponse:
    """Mock Solr select response."""
    return SolrSelectResponse(
        response=SolrSelectResponseBody(
            docs=mock_solr_response_docs,
            num_found=len(mock_solr_response_docs),
            num_found_exact=True,
            start=0,
        ),
        response_header=SolrResponseHeader(status=200),
    )


@pytest.fixture
def mock_vector_store_query_result(
    mock_solr_response_docs: list[dict[str, Any]],
) -> VectorStoreQueryResult:
    """Mock vector store query result."""
    return VectorStoreQueryResult(
        ids=[doc["id"] for doc in mock_solr_response_docs],
        nodes=[
            TextNode(
                id_=doc["id"],
                text=doc["contents"],
                embedding=doc["embedding"],
                metadata={
                    "extra_field": doc["extra_field"],
                    "other_extra_field": doc["other_extra_field"],
                },
            )
            for doc in mock_solr_response_docs
        ],
        similarities=[doc["score"] for doc in mock_solr_response_docs],
    )


@pytest.fixture
def mock_solr_vector_store(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
) -> ApacheSolrVectorStore:
    """Mock Solr vector store with basic configuration."""
    return ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        docid_field="docid",
        content_field="contents",
        embedding_field="embedding",
        metadata_to_solr_field_mapping=[("author", "author_field")],
        solr_field_preprocessor_kwargs={},
    )


def create_sample_input_nodes(
    num_nodes: int = 3,
) -> tuple[list[BaseNode], list[dict[str, Any]]]:
    """Create sample input nodes and expected Solr data for testing."""
    nodes = []
    expected_data = []

    for i in range(num_nodes):
        node = TextNode(
            id_=f"node{i}",
            text=f"content {i}",
            embedding=[0.1 * i, 0.2 * i, 0.3 * i],
            metadata={"author": f"author{i}", "topic": f"topic{i}"},
        )
        nodes.append(node)

        expected_data.append(
            {
                "id": f"node{i}",
                "contents": f"content {i}",
                "embedding": [0.1 * i, 0.2 * i, 0.3 * i],
                "docid": None,
                "author_field": f"author{i}",  # mapped via metadata_to_solr_field_mapping
            }
        )

    return nodes, expected_data


@pytest.mark.parametrize(
    ("query_dict", "expected_fields"),
    [
        # Minimal required fields only
        (
            {"q": "*:*", "fq": []},
            {"q": "*:*", "fq": []},
        ),
        # All fields present
        (
            {
                "q": "{!knn f=embedding topK=10}[0.1, 0.2, 0.3]",
                "fq": ["field1:value1", "field2:value2"],
                "fl": "id,content,score",
                "rows": "20",
            },
            {
                "q": "{!knn f=embedding topK=10}[0.1, 0.2, 0.3]",
                "fq": ["field1:value1", "field2:value2"],
                "fl": "id,content,score",
                "rows": "20",
            },
        ),
        # With optional fl only
        (
            {"q": "title:test", "fq": ["status:active"], "fl": "id,title"},
            {"q": "title:test", "fq": ["status:active"], "fl": "id,title"},
        ),
        # With optional rows only
        (
            {"q": "content:search", "fq": [], "rows": "50"},
            {"q": "content:search", "fq": [], "rows": "50"},
        ),
    ],
    ids=[
        "Minimal required fields",
        "All fields present",
        "With optional fl",
        "With optional rows",
    ],
)
def test_solr_query_dict_successful_creation(
    query_dict: dict[str, Any], expected_fields: dict[str, Any]
) -> None:
    """Test successful creation of SolrQueryDict with various field combinations."""
    solr_query: SolrQueryDict = query_dict

    for key, expected_value in expected_fields.items():
        assert solr_query[key] == expected_value


"""Store Creation Tests"""


@pytest.mark.parametrize(
    ("additional_config", "expected_attributes"),
    [
        # Minimal configuration
        ({}, {"docid_field": "docid", "content_field": "contents"}),
        # Override some fields
        (
            {"content_field": "text_content", "embedding_field": "vectors"},
            {"content_field": "text_content", "embedding_field": "vectors"},
        ),
        # Add text search fields
        (
            {"text_search_fields": [BoostedTextField(field="title", boost_factor=2.0)]},
            {"text_search_fields": [BoostedTextField(field="title", boost_factor=2.0)]},
        ),
        # Custom output fields
        (
            {"output_fields": ["id", "title"]},
            {"output_fields": ["id", "title", "score"]},  # score auto-added
        ),
    ],
    ids=[
        "Default fixture config",
        "Override fields",
        "With text search",
        "Custom output fields",
    ],
)
def test_vector_store_successful_creation(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    additional_config: dict[str, Any],
    expected_attributes: dict[str, Any],
) -> None:
    """Test successful creation of ApacheSolrVectorStore using existing fixtures."""
    base_config = {
        "sync_client": mock_sync_client,
        "async_client": mock_async_client,
        "nodeid_field": "id",
        "docid_field": "docid",
        "content_field": "contents",
        "embedding_field": "embedding",
        "metadata_to_solr_field_mapping": [("author", "author_field")],
        "solr_field_preprocessor_kwargs": {},
    }

    store = ApacheSolrVectorStore(**{**base_config, **additional_config})

    assert store.sync_client is mock_sync_client
    assert store.async_client is mock_async_client
    assert store.nodeid_field == "id"

    for attr_name, expected_value in expected_attributes.items():
        assert getattr(store, attr_name) == expected_value


@pytest.mark.parametrize(
    ("invalid_config", "error_match"),
    [
        # Missing required sync_client
        (
            {
                "async_client": AsyncMock(),
                "nodeid_field": "id",
                "solr_field_preprocessor_kwargs": {},
            },
            "Field required",
        ),
        # Missing required async_client
        (
            {
                "sync_client": MagicMock(),
                "nodeid_field": "id",
                "solr_field_preprocessor_kwargs": {},
            },
            "Field required",
        ),
        # Missing required nodeid_field
        (
            {
                "sync_client": MagicMock(),
                "async_client": AsyncMock(),
                "solr_field_preprocessor_kwargs": {},
            },
            "Field required",
        ),
        # Empty text_search_fields (violates MinLen(1))
        (
            {
                "sync_client": MagicMock(),
                "async_client": AsyncMock(),
                "nodeid_field": "id",
                "text_search_fields": [],
                "solr_field_preprocessor_kwargs": {},
            },
            "at least 1",
        ),
        # Empty output_fields (violates MinLen(1))
        (
            {
                "sync_client": MagicMock(),
                "async_client": AsyncMock(),
                "nodeid_field": "id",
                "output_fields": [],
                "solr_field_preprocessor_kwargs": {},
            },
            "at least 1",
        ),
    ],
    ids=[
        "Missing sync_client",
        "Missing async_client",
        "Missing nodeid_field",
        "Empty text_search_fields",
        "Empty output_fields",
    ],
)
def test_apache_solr_vector_store_creation_failures(
    invalid_config: dict[str, Any],
    error_match: str,
) -> None:
    """Test ApacheSolrVectorStore creation validation failures."""
    with pytest.raises(ValueError, match=error_match):
        ApacheSolrVectorStore(**invalid_config)


def test_vector_store_output_fields_validation(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
) -> None:
    """Test that output_fields validator ensures 'score' is always included."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        output_fields=["field1", "field2"],
        solr_field_preprocessor_kwargs={},
    )

    #'score' should be automatically added
    assert "score" in store.output_fields
    assert set(store.output_fields) == {"field1", "field2", "score"}

    # output_fields already with 'score'
    store2 = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        output_fields=["field1", "score", "field2"],
        solr_field_preprocessor_kwargs={},
    )

    #'score' should not be duplicated
    assert store2.output_fields.count("score") == 1


def test_vector_store_client_property(
    mock_solr_vector_store: ApacheSolrVectorStore,
    mock_sync_client: MagicMock,
) -> None:
    """Test client property returns sync client."""
    actual_client = mock_solr_vector_store.client
    assert actual_client is mock_sync_client


def test_vector_store_aclient_property(
    mock_solr_vector_store: ApacheSolrVectorStore,
    mock_async_client: AsyncMock,
) -> None:
    """Test aclient property returns async client."""
    actual_client = mock_solr_vector_store.aclient
    assert actual_client is mock_async_client


"""Query Construction Tests"""


@pytest.mark.parametrize(
    (
        "store_embedding_field",
        "query_embedding_field",
        "query_embedding",
        "similarity_top_k",
        "should_succeed",
    ),
    [
        # Success cases
        ("store_embedding", None, [1, 2, 3], None, True),
        (None, "query_embedding", [1, 2, 3], None, True),
        ("store_embedding", "query_embedding", [1, 2, 3], 10, True),
        # Failure cases
        (None, None, [1, 2, 3], None, False),  # No embedding field specified
    ],
    ids=[
        "store_field_only_success",
        "query_field_only_success",
        "both_fields_success",
        "no_field_fail",
    ],
)
def test_to_solr_query_dense(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    store_embedding_field: Optional[str],
    query_embedding_field: Optional[str],
    query_embedding: Optional[list[float]],
    similarity_top_k: Optional[int],
    should_succeed: bool,
) -> None:
    """Test _to_solr_query for dense vector queries - both success and failure cases."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        embedding_field=store_embedding_field,
        solr_field_preprocessor_kwargs={},
    )

    query_args = {
        "mode": VectorStoreQueryMode.DEFAULT,
        "query_embedding": query_embedding,
    }
    if query_embedding_field:
        query_args["embedding_field"] = query_embedding_field
    if similarity_top_k:
        query_args["similarity_top_k"] = similarity_top_k

    query = VectorStoreQuery(**query_args)

    if should_succeed:
        # success case
        result = store._to_solr_query(query)

        # Verify KNN query structure
        assert "{!knn f=" in result["q"]
        assert result["q"].endswith(f"{query_embedding}")

        # Check field precedence: query field > store field
        expected_field = query_embedding_field or store_embedding_field
        assert f"f={expected_field}" in result["q"]

        # Check topK value
        expected_topk = similarity_top_k or 1
        assert f"topK={expected_topk}" in result["q"]
    else:
        # failure case
        with pytest.raises(ValueError):
            store._to_solr_query(query)


@pytest.mark.parametrize(
    ("text_search_fields", "query_str", "sparse_top_k", "should_succeed"),
    [
        # Success cases
        (
            [BoostedTextField(field="title", boost_factor=2.0)],
            "test query",
            None,
            True,
        ),
        (
            [BoostedTextField(field="title"), BoostedTextField(field="content")],
            "test",
            20,
            True,
        ),
        # Failure cases
        (None, "test query", None, False),  # No text search fields
        ([BoostedTextField(field="title")], None, None, False),  # No query string
    ],
    ids=[
        "single_field_success",
        "multiple_fields_success",
        "no_fields_fail",
        "no_query_str_fail",
    ],
)
def test_to_solr_query_bm25(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    text_search_fields: Optional[list[BoostedTextField]],
    query_str: Optional[str],
    sparse_top_k: Optional[int],
    should_succeed: bool,
) -> None:
    """Test _to_solr_query for BM25 text search queries - both success and failure cases."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        text_search_fields=text_search_fields,
        solr_field_preprocessor_kwargs={},
    )

    query_args = {
        "mode": VectorStoreQueryMode.TEXT_SEARCH,
        "query_str": query_str,
    }
    if sparse_top_k:
        query_args["sparse_top_k"] = sparse_top_k

    query = VectorStoreQuery(**query_args)

    if should_succeed:
        # success case
        result = store._to_solr_query(query)

        # verify dismax query structure
        assert "{!dismax" in result["q"]
        assert "qf=" in result["q"]
        # Check for escaped query string (spaces become \\ )
        escaped_query_str = query_str.replace(" ", "\\ ")
        assert escaped_query_str in result["q"]

        # Check field boosting is preserved
        for field in text_search_fields:
            field_str = field.get_query_str()
            assert field_str in result["q"]

        # Check rows parameter - only if sparse_top_k was provided
        if sparse_top_k is not None:
            assert result["rows"] == str(sparse_top_k)
        else:
            # When sparse_top_k is None, rows should not be set
            assert result["rows"] == "None"
    else:
        # failure case
        with pytest.raises(ValueError):
            store._to_solr_query(query)


@pytest.mark.parametrize(
    ("doc_ids", "node_ids", "filters", "output_fields"),
    [
        # Test various combinations of optional parameters
        (["doc1", "doc2"], None, None, None),
        (None, ["node1", "node2"], None, ["field1", "field2"]),
        (
            None,
            None,
            MetadataFilters(filters=[MetadataFilter(key="status", value="active")]),
            None,
        ),
        (
            ["doc1"],
            ["node1"],
            MetadataFilters(filters=[MetadataFilter(key="type", value="article")]),
            ["id", "title"],
        ),
    ],
    ids=[
        "doc_ids_only",
        "node_ids_and_output_fields",
        "filters_only",
        "all_optional_params",
    ],
)
def test_to_solr_query_optional_params(
    mock_solr_vector_store: ApacheSolrVectorStore,
    doc_ids: Optional[list[str]],
    node_ids: Optional[list[str]],
    filters: Optional[MetadataFilters],
    output_fields: Optional[list[str]],
) -> None:
    """Test _to_solr_query handles optional parameters correctly."""
    query = VectorStoreQuery(
        mode=VectorStoreQueryMode.DEFAULT,
        query_embedding=[0.1, 0.2, 0.3],
        doc_ids=doc_ids,
        node_ids=node_ids,
        filters=filters,
        output_fields=output_fields,
    )

    result = mock_solr_vector_store._to_solr_query(query)

    # Check filter queries (fq) are built correctly
    if doc_ids:
        doc_fq = f"docid:({' OR '.join(doc_ids)})"
        assert doc_fq in result["fq"]

    if node_ids:
        node_fq = f"id:({' OR '.join(node_ids)})"
        assert node_fq in result["fq"]

    if filters:
        # At least one filter should be present in fq
        assert len(result["fq"]) > 0

    # Check field list (fl) parameter
    if output_fields:
        expected_fl = ",".join([*output_fields, "score"])
        assert result["fl"] == expected_fl
    else:
        assert result["fl"] == "*,score"


def test_to_solr_query_docid_field_missing_error(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
) -> None:
    """Test _to_solr_query raises error when doc_ids provided but docid_field is None."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        docid_field=None,  # No docid field configured
        embedding_field="embedding",
        solr_field_preprocessor_kwargs={},
    )

    query = VectorStoreQuery(
        mode=VectorStoreQueryMode.DEFAULT,
        query_embedding=[0.1, 0.2, 0.3],
        doc_ids=["doc1", "doc2"],  # Trying to filter by doc_ids
    )

    with pytest.raises(ValueError, match="`docid_field` must be passed"):
        store._to_solr_query(query)


"""Query Execution Tests"""


@pytest.mark.parametrize(
    "content_field", ["contents", None], ids=["Has contents", "No contents"]
)
@pytest.mark.parametrize(
    "embedding_field", ["embedding", None], ids=["Has embedding", "No embedding"]
)
def test_vector_store_process_query_results(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    mock_solr_response_docs: list[dict[str, Any]],
    content_field: Optional[str],
    embedding_field: Optional[str],
) -> None:
    """Test _process_query_results method."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        docid_field="docid",
        content_field=content_field,
        embedding_field=embedding_field,
        metadata_to_solr_field_mapping=None,
        solr_field_preprocessor_kwargs={},
    )

    # Prepare test data
    results = mock_solr_response_docs.copy()
    if not embedding_field:
        for doc in results:
            del doc["embedding"]

    # Prepare expected metadata
    metadata: dict[str, Any] = {
        "extra_field": "extra field",
        "other_extra_field": "other extra field",
    }
    if not content_field:
        metadata["contents"] = "some text"

    expected_results = VectorStoreQueryResult(
        ids=["node0", "node1"],
        nodes=[
            TextNode(
                id_="node0",
                text="some text" if content_field else "",
                embedding=[0.1, 0.2, 0.3] if embedding_field else None,
                metadata=metadata,
            ),
            TextNode(
                id_="node1",
                text="some text" if content_field else "",
                embedding=[0.1, 0.2, 0.3] if embedding_field else None,
                metadata=metadata,
            ),
        ],
        similarities=[0.95, 0.85],
    )

    actual_results = store._process_query_results(results)

    assert actual_results == expected_results


@pytest.mark.parametrize(
    "mode",
    [VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID],
    ids=["Hybrid Mode", "Semantic Hybrid Mode"],
)
def test_vector_store_validate_query_mode_unsupported(
    mock_solr_vector_store: ApacheSolrVectorStore,
    mode: VectorStoreQueryMode,
) -> None:
    """Test that unsupported query modes raise ValueError."""
    query = VectorStoreQuery(mode=mode)

    with pytest.raises(ValueError, match="ApacheSolrVectorStore does not support"):
        mock_solr_vector_store._validate_query_mode(query)


def test_vector_store_query(
    mock_solr_vector_store: ApacheSolrVectorStore,
    mock_solr_response: SolrSelectResponse,
    mock_vector_store_query_result: VectorStoreQueryResult,
) -> None:
    """Test synchronous query method."""
    input_query = VectorStoreQuery(
        embedding_field="embedding",
        query_embedding=[0.1, 0.2, 0.3],
        output_fields=None,
    )
    mock_solr_vector_store.sync_client.search.return_value = mock_solr_response

    actual_results = mock_solr_vector_store.query(input_query)

    assert actual_results == mock_vector_store_query_result
    mock_solr_vector_store.sync_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_aquery(
    mock_solr_vector_store: ApacheSolrVectorStore,
    mock_solr_response: SolrSelectResponse,
    mock_vector_store_query_result: VectorStoreQueryResult,
) -> None:
    """Test asynchronous query method."""
    input_query = VectorStoreQuery(
        embedding_field="embedding",
        query_embedding=[0.1, 0.2, 0.3],
        output_fields=None,
    )
    mock_solr_vector_store.async_client.search.return_value = mock_solr_response

    actual_results = await mock_solr_vector_store.aquery(input_query)

    assert actual_results == mock_vector_store_query_result
    mock_solr_vector_store.async_client.search.assert_called_once()


@pytest.mark.parametrize(
    ("nodeid_field", "nodeid_data"),
    [("id", {"id": "node1"})],
    ids=["Has nodeid"],
)
@pytest.mark.parametrize(
    ("content_field", "content_data"),
    [("contents", {"contents": "some text"}), (None, {})],
    ids=["Has contents", "No contents"],
)
@pytest.mark.parametrize(
    ("embedding_field", "embedding_data"),
    [("embedding", {"embedding": [1, 2, 3]}), (None, {})],
    ids=["Has embedding", "No embedding"],
)
@pytest.mark.parametrize(
    ("docid_field", "docid_data"),
    [("docid", {"docid": "doc1"}), (None, {})],
    ids=["Has docid", "No docid"],
)
@pytest.mark.parametrize(
    ("metadata_to_solr_field_mapping", "solr_field_data"),
    [
        (
            [
                ("doc_field1", "solr_field1"),
                ("doc_field2", "solr_field2"),
                ("doc_missing_field", "solr_missing_field"),
            ],
            {"solr_field1": "v1", "solr_field2": "v2"},
        ),
        (None, {}),
    ],
    ids=["Has metadata_to_solr_field_mapping", "No metadata_to_solr_field_mapping"],
)
def test_apache_solr_vector_store_get_data_from_node(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    nodeid_field: Optional[str],
    nodeid_data: dict[str, Any],
    content_field: Optional[str],
    content_data: dict[str, Any],
    embedding_field: Optional[str],
    embedding_data: dict[str, Any],
    docid_field: Optional[str],
    docid_data: dict[str, Any],
    metadata_to_solr_field_mapping: Optional[list[tuple[str, str]]],
    solr_field_data: dict[str, Any],
) -> None:
    """Test _get_data_from_node method with various field configurations."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field=nodeid_field,
        docid_field=docid_field,
        content_field=content_field,
        embedding_field=embedding_field,
        metadata_to_solr_field_mapping=metadata_to_solr_field_mapping,
        solr_field_preprocessor_kwargs={},
    )
    expected_data = {
        **nodeid_data,
        **content_data,
        **embedding_data,
        **docid_data,
        **solr_field_data,
    }

    # Mock node
    input_node = MagicMock(
        node_id="node1",
        get_content=MagicMock(return_value="some text"),
        get_embedding=MagicMock(return_value=[1, 2, 3]),
        ref_doc_id="doc1",
        metadata={"doc_field1": "v1", "doc_field2": "v2"},
    )

    actual_data = store._get_data_from_node(input_node)

    assert actual_data == expected_data


def test_apache_solr_vector_store_get_data_from_nodes_valid(
    mock_solr_vector_store: ApacheSolrVectorStore,
) -> None:
    """Test _get_data_from_nodes method."""
    input_nodes, expected_data = create_sample_input_nodes()
    expected_ids = [node.id_ for node in input_nodes]

    actual_ids, actual_data = mock_solr_vector_store._get_data_from_nodes(input_nodes)

    assert actual_ids == expected_ids
    assert actual_data == expected_data


"""Add and Delete Tests"""


@params_add_kwargs
def test_vector_store_add(
    mock_solr_vector_store: ApacheSolrVectorStore,
    add_kwargs: dict[str, Any],
) -> None:
    """Test synchronous add method."""
    input_nodes, expected_data = create_sample_input_nodes()
    expected_ids = [node.id_ for node in input_nodes]

    actual_ids = mock_solr_vector_store.add(input_nodes, **add_kwargs)

    assert actual_ids == expected_ids
    mock_solr_vector_store.sync_client.add.assert_called_once_with(expected_data)


@params_add_kwargs
@pytest.mark.asyncio
async def test_vector_store_async_add(
    mock_solr_vector_store: ApacheSolrVectorStore,
    add_kwargs: dict[str, Any],
) -> None:
    """Test asynchronous add method."""
    input_nodes, expected_data = create_sample_input_nodes()
    expected_ids = [node.id_ for node in input_nodes]

    actual_ids = await mock_solr_vector_store.async_add(input_nodes, **add_kwargs)

    assert actual_ids == expected_ids
    mock_solr_vector_store.async_client.add.assert_called_once_with(expected_data)


@pytest.mark.asyncio
async def test_vector_store_async_add_raises_for_empty_node_list(
    mock_solr_vector_store: ApacheSolrVectorStore,
) -> None:
    """Test async_add raises error for empty node list."""
    with pytest.raises(ValueError, match="Call to 'async_add' with no contents"):
        await mock_solr_vector_store.async_add([])


@params_delete_kwargs
def test_vector_store_delete(
    mock_solr_vector_store: ApacheSolrVectorStore,
    delete_kwargs: dict[str, Any],
) -> None:
    """Test synchronous delete method."""
    input_ref_doc_id = "doc1"

    mock_solr_vector_store.delete(input_ref_doc_id, **delete_kwargs)

    mock_solr_vector_store.sync_client.delete_by_id.assert_called_once_with(
        [input_ref_doc_id]
    )


def test_vector_store_delete_no_docid_field(
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
) -> None:
    """Test delete works even when docid_field is None."""
    store = ApacheSolrVectorStore(
        sync_client=mock_sync_client,
        async_client=mock_async_client,
        nodeid_field="id",
        docid_field=None,
        solr_field_preprocessor_kwargs={},
    )

    store.delete("doc1")

    # delete_by_id should still be called with the ref_doc_id
    mock_sync_client.delete_by_id.assert_called_once_with(["doc1"])


@params_delete_kwargs
@pytest.mark.asyncio
async def test_vector_store_adelete(
    mock_solr_vector_store: ApacheSolrVectorStore,
    delete_kwargs: dict[str, Any],
) -> None:
    """Test asynchronous delete method."""
    input_ref_doc_id = "doc1"

    await mock_solr_vector_store.adelete(input_ref_doc_id, **delete_kwargs)

    mock_solr_vector_store.async_client.delete_by_id.assert_called_once_with(
        [input_ref_doc_id]
    )


# Parameters for delete_nodes / adelete_nodes that use delete_by_id (no filters)
@pytest.mark.parametrize(
    (
        "input_nodes",
        "input_filters",
    ),
    [
        (["node1", "node2"], None),
        (["node1", "node2"], MetadataFilters(filters=[])),
    ],
    ids=[
        "node_ids=non-empty, filters=None",
        "node_ids=non-empty, filters=empty set",
    ],
)
@params_delete_kwargs
def test_vector_store_delete_nodes_by_id(
    mock_solr_vector_store: ApacheSolrVectorStore,
    input_nodes: Optional[list[str]],
    input_filters: Optional[MetadataFilters],
    delete_kwargs: dict[str, Any],
) -> None:
    """Test synchronous delete_nodes method using delete_by_id."""
    mock_solr_vector_store.delete_nodes(input_nodes, input_filters, **delete_kwargs)

    mock_solr_vector_store.sync_client.delete_by_id.assert_called_once_with(input_nodes)


# Parameters for delete_nodes / adelete_nodes that use delete_by_query (with filters)
@pytest.mark.parametrize(
    (
        "input_nodes",
        "input_filters",
        "expected_query",
    ),
    [
        (
            None,
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="docid", value="doc1", operator=FilterOperator.EQ
                    )
                ]
            ),
            "((docid:doc1))",
        ),
        (
            [],
            MetadataFilters(filters=[MetadataFilter(key="docid", value="doc1")]),
            "((docid:doc1))",
        ),
        (
            None,
            MetadataFilters(
                filters=[
                    MetadataFilter(key="docid", value="doc1"),
                    MetadataFilter(key="docid", value="doc2"),
                ],
                condition=FilterCondition.OR,
            ),
            "((docid:doc1) OR (docid:doc2))",
        ),
        (
            ["node1", "node2"],
            MetadataFilters(filters=[MetadataFilter(key="docid", value="doc1")]),
            "(id:(node1 OR node2) AND ((docid:doc1)))",
        ),
    ],
    ids=[
        "node_ids=None, filters=1 filter",
        "node_ids=[], filters=1 filter",
        "node_ids=None, filters=2 filters",
        "node_ids=non-empty, filters=1 filter",
    ],
)
@params_delete_kwargs
def tes_vector_store_delete_nodes(
    mock_solr_vector_store: ApacheSolrVectorStore,
    input_nodes: Optional[list[str]],
    input_filters: Optional[MetadataFilters],
    expected_query: str,
    delete_kwargs: dict[str, Any],
) -> None:
    """Test synchronous delete_nodes method using delete_by_query."""
    mock_solr_vector_store.delete_nodes(input_nodes, input_filters, **delete_kwargs)

    mock_solr_vector_store.sync_client.delete_by_query.assert_called_once_with(
        expected_query
    )


# Parameters for adelete_nodes that use delete_by_id (no filters)
@pytest.mark.parametrize(
    (
        "input_nodes",
        "input_filters",
    ),
    [
        (["node1", "node2"], None),
    ],
    ids=[
        "node_ids=non-empty, filters=None",
    ],
)
@params_delete_kwargs
@pytest.mark.asyncio
async def test_vector_store_adelete_nodes_by_id(
    mock_solr_vector_store: ApacheSolrVectorStore,
    input_nodes: Optional[list[str]],
    input_filters: Optional[MetadataFilters],
    delete_kwargs: dict[str, Any],
) -> None:
    """Test asynchronous delete_nodes method using delete_by_id."""
    await mock_solr_vector_store.adelete_nodes(
        input_nodes, input_filters, **delete_kwargs
    )

    mock_solr_vector_store.async_client.delete_by_id.assert_called_once_with(
        input_nodes
    )


# Parameters for adelete_nodes that use delete_by_query (with filters)
@pytest.mark.parametrize(
    (
        "input_nodes",
        "input_filters",
        "expected_query",
    ),
    [
        (
            None,
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="docid", value="doc1", operator=FilterOperator.EQ
                    )
                ]
            ),
            "((docid:doc1))",
        ),
    ],
    ids=[
        "node_ids=None, filters=1 filter",
    ],
)
@params_delete_kwargs
@pytest.mark.asyncio
async def test_vector_store_adelete_nodes(
    mock_solr_vector_store: ApacheSolrVectorStore,
    input_nodes: Optional[list[str]],
    input_filters: Optional[MetadataFilters],
    expected_query: str,
    delete_kwargs: dict[str, Any],
) -> None:
    """Test asynchronous delete_nodes method using delete_by_query."""
    await mock_solr_vector_store.adelete_nodes(
        input_nodes, input_filters, **delete_kwargs
    )

    mock_solr_vector_store.async_client.delete_by_query.assert_called_once_with(
        expected_query
    )


@pytest.mark.parametrize(
    ("input_nodes", "input_filters", "error_match"),
    [
        (None, None, "At least one of `node_ids` or `filters` must be passed"),
        ([], None, "At least one of `node_ids` or `filters` must be passed"),
        (
            None,
            MetadataFilters(filters=[]),
            "Neither `node_ids` nor non-empty `filters` were passed",
        ),
        (
            [],
            MetadataFilters(filters=[]),
            "Neither `node_ids` nor non-empty `filters` were passed",
        ),
    ],
    ids=[
        "node_ids=None, filters=None",
        "node_ids=[], filters=None",
        "node_ids=None, filters=empty filter",
        "node_ids=[], filters=empty filter",
    ],
)
@params_delete_kwargs
def test_vector_store_delete_nodes_invalid_input(
    mock_solr_vector_store: ApacheSolrVectorStore,
    input_nodes: Optional[list[str]],
    input_filters: Optional[MetadataFilters],
    delete_kwargs: dict[str, Any],
    error_match: str,
) -> None:
    """Test delete_nodes raises error for invalid input."""
    with pytest.raises(ValueError, match=error_match):
        mock_solr_vector_store.delete_nodes(input_nodes, input_filters, **delete_kwargs)


"""Cleanup Tests"""


def test_vector_store_clear(
    mock_solr_vector_store: ApacheSolrVectorStore,
) -> None:
    """Test synchronous clear method."""
    mock_solr_vector_store.clear()

    mock_solr_vector_store.sync_client.clear_collection.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_aclear(
    mock_solr_vector_store: ApacheSolrVectorStore,
) -> None:
    """Test asynchronous clear method."""
    await mock_solr_vector_store.aclear()

    mock_solr_vector_store.async_client.clear_collection.assert_called_once()
