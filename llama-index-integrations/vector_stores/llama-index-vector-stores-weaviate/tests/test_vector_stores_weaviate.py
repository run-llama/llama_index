from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.vector_stores.weaviate import (
    WeaviateVectorStore,
    SyncClientNotProvidedError,
    AsyncClientNotProvidedError,
)
import asyncio
import pytest
import pytest_asyncio
import weaviate
import weaviate.classes as wvc

import weaviate.embedded
from unittest.mock import MagicMock, AsyncMock
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from weaviate.collections.batch.base import (
    _DynamicBatching,
    _FixedSizeBatching,
)

TEST_COLLECTION_NAME = "TestCollection"


def test_no_weaviate_client_instance_provided():
    """Tests that the creation of a Weaviate client within the WeaviateVectorStore constructor works."""
    vector_store = WeaviateVectorStore(
        client_kwargs={"embedded_options": weaviate.embedded.EmbeddedOptions()}
    )

    # Make sure that the vector store is functional by calling some basic methods
    vector_store.add([TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])])
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 1
    weaviate_client = vector_store.client
    del vector_store
    assert not weaviate_client.is_connected()  # As the Weaviate client was created within WeaviateVectorStore, it lies in its responsibility to close the connection when it is not longer needed


# Async Tests


@pytest.fixture(scope="module")
def launcher(tmp_path_factory):
    """Starts the embedded Weaviate instance once for the module."""
    # Create a temporary directory for this module's Weaviate instance
    data_path = tmp_path_factory.mktemp("weaviate_launcher")
    client = weaviate.connect_to_embedded(
        port=8079, grpc_port=50050, persistence_data_path=str(data_path)
    )
    yield client
    client.close()


@pytest_asyncio.fixture(scope="module")
async def async_client(launcher):
    """Connects to the already running embedded instance."""
    client = weaviate.use_async_with_local(port=8079, grpc_port=50050)
    await client.connect()
    yield client
    await client.close()


# This replaces the event loop which is deprecated (discussion: https://github.com/pytest-dev/pytest-asyncio/discussions/587)
# It was necessary to implement it this way due to pytest 7 currently always being used in the pants test performed during CI.
# TODO Revert the commit where it was implemented like this as soon as pytest >= 8 is used in the pants tests (discussion: https://github.com/run-llama/llama_index/pull/17220#issuecomment-2532175072)
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_vector_store(async_client, event_loop):
    vector_store = WeaviateVectorStore(
        weaviate_client=async_client, index_name=TEST_COLLECTION_NAME
    )
    # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
    await vector_store.aclear()
    yield vector_store
    await vector_store.aclear()


@pytest.mark.asyncio
async def test_async_basic_flow(async_vector_store):
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]

    await async_vector_store.async_add(nodes)

    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = await async_vector_store.aquery(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "This is a test."
    assert results.similarities[0] == 1.0

    assert results.similarities[0] > results.similarities[1]


@pytest.mark.asyncio
async def test_async_old_data_gone(async_vector_store):
    """Makes sure that no data stays in the database in between tests (otherwise more than one node would be found in the assertion)."""
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
    ]

    await async_vector_store.async_add(nodes)

    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=2,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = await async_vector_store.aquery(query)

    assert len(results.nodes) == 1


@pytest.mark.asyncio
async def test_async_delete_nodes(async_vector_store):
    node_to_be_deleted = TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])
    node_to_keep = TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0])
    nodes = [node_to_be_deleted, node_to_keep]

    await async_vector_store.async_add(nodes)
    await async_vector_store.adelete_nodes(node_ids=[node_to_be_deleted.node_id])
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == node_to_keep.node_id


@pytest.mark.asyncio
async def test_async_delete(async_vector_store):
    node_to_be_deleted = TextNode(
        text="Hello world.",
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_deleted")
        },
        embedding=[0.0, 0.0, 0.3],
    )
    node_to_keep = TextNode(
        text="This is a test.",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_kept")},
        embedding=[0.3, 0.0, 0.0],
    )
    nodes = [node_to_be_deleted, node_to_keep]
    await async_vector_store.async_add(nodes)

    # First check that nothing gets deleted if no matching nodes are present
    await async_vector_store.adelete(ref_doc_id="no_match_in_db")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 2

    # Now test actual deletion
    await async_vector_store.adelete(ref_doc_id="to_be_deleted")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == node_to_keep.node_id


@pytest.mark.asyncio
async def test_async_client_properties(async_vector_store):
    assert isinstance(async_vector_store.async_client, weaviate.WeaviateAsyncClient)
    with pytest.raises(SyncClientNotProvidedError):
        async_vector_store.client


# Sync Tests


def test_class():
    names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture(scope="module")
def client(launcher):
    """Reuses the launcher client for sync tests."""
    return launcher


@pytest.fixture()
def vector_store(client):
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=TEST_COLLECTION_NAME
    )
    vector_store.clear()  # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
    yield vector_store
    vector_store.clear()


@pytest.fixture()
def vector_store_with_sample_nodes(vector_store):
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]
    vector_store.add(nodes)
    return vector_store


def test_vector_store_with_custom_batch(client):
    # default, dynamic batch
    vector_store_default_dynamic = WeaviateVectorStore(
        weaviate_client=client, index_name=TEST_COLLECTION_NAME
    )
    assert isinstance(client.batch._batch_mode, _DynamicBatching)
    # custom, with fixed size
    custom_batch = client.batch.fixed_size(
        batch_size=123,
        concurrent_requests=3,
        consistency_level=weaviate.classes.config.ConsistencyLevel.ONE,
    )
    vector_store_fixed = WeaviateVectorStore(
        weaviate_client=client,
        index_name=TEST_COLLECTION_NAME,
        client_kwargs={"custom_batch": custom_batch},
    )
    assert isinstance(client.batch._batch_mode, _FixedSizeBatching)
    assert client.batch._batch_mode.batch_size == 123
    assert client.batch._batch_mode.concurrent_requests == 3
    assert (
        client.batch._consistency_level == weaviate.classes.config.ConsistencyLevel.ONE
    )

    vector_store_default_dynamic.clear()
    vector_store_fixed.clear()

    # test wrong value
    try:
        WeaviateVectorStore(
            weaviate_client=client,
            index_name=TEST_COLLECTION_NAME,
            client_kwargs={"custom_batch": "wrong_value"},
        )
        pytest.fail("ValueError not raised for invalid custom_batch value")
    except ValueError:
        pass


def test_sync_basic_flow(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="world",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = vector_store_with_sample_nodes.query(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "This is a test."
    assert results.similarities[0] == 1.0

    assert results.similarities[0] > results.similarities[1]


def test_hybrid_search(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=10,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store_with_sample_nodes.query(query)
    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello world."
    assert results.nodes[1].text == "This is a test."

    assert results.similarities[0] > results.similarities[1]


def test_query_kwargs(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store_with_sample_nodes.query(
        query,
        max_vector_distance=0.0,
    )
    assert len(results.nodes) == 0


def test_can_query_collection_with_complex_property_types(client):
    """Verifies that it is possible to query data from collections that contain complex properties (e.g. a list of nested objects in one of the properties)."""
    collection_name = "ComplexTypeInArrayTest"
    client.collections.delete(collection_name)
    collection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="array_prop",
                data_type=wvc.config.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wvc.config.Property(
                        name="nested_prop",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                ],
            ),
        ],
    )

    collection.data.insert(
        {
            "text": "Text of object containing complex properties",
            "array_prop": [{"nested_prop": "nested_prop content"}],
        },
        vector=[1.0, 0.0, 0.0],
    )

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=collection_name,
    )
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = vector_store.query(query)

    assert len(results.nodes) == 1
    assert results.nodes[0].text == "Text of object containing complex properties"


def test_sync_delete(vector_store):
    node_to_be_deleted = TextNode(
        text="Hello world.",
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_deleted")
        },
        embedding=[0.0, 0.0, 0.3],
    )
    node_to_keep = TextNode(
        text="This is a test.",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_kept")},
        embedding=[0.3, 0.0, 0.0],
    )
    nodes = [node_to_be_deleted, node_to_keep]
    vector_store.add(nodes)

    # First check that nothing gets deleted if no matching nodes are present
    vector_store.delete(ref_doc_id="no_match_in_db")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 2

    # Now test actual deletion
    vector_store.delete(ref_doc_id="to_be_deleted")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 1
    results.nodes[0].node_id == node_to_keep.node_id


def test_no_weaviate_client_instance_provided(tmp_path):
    """Tests that the creation of a Weaviate client within the WeaviateVectorStore constructor works."""
    # Use different ports and data path to avoid conflict with the shared launcher
    data_path = tmp_path / "weaviate_no_client"
    vector_store = WeaviateVectorStore(
        client_kwargs={
            "embedded_options": weaviate.embedded.EmbeddedOptions(
                port=8066, grpc_port=50066, persistence_data_path=str(data_path)
            )
        }
    )

    # Make sure that the vector store is functional by calling some basic methods
    vector_store.add([TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])])
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 1
    weaviate_client = vector_store.client
    del vector_store
    assert not weaviate_client.is_connected()  # As the Weaviate client was created within WeaviateVectorStore, it lies in its responsibility to close the connection when it is not longer needed


@pytest.mark.asyncio
async def test_async_methods_called_without_async_client(vector_store):
    """Makes sure that we present an easy to understand error message to the user if he did not not provide an async client, but tried to call async methods."""
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.async_add(
            [TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])]
        )
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.adelete(ref_doc_id="no_match_in_db")
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.adelete_nodes(node_ids=["sample_node_id"])
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.aclear()
    with pytest.raises(AsyncClientNotProvidedError):
        query = VectorStoreQuery(
            query_embedding=[0.3, 0.0, 0.0],
            similarity_top_k=10,
            query_str="test",
            mode=VectorStoreQueryMode.DEFAULT,
        )
        await vector_store.aquery(query)


def test_sync_client_properties(vector_store):
    assert isinstance(vector_store.client, weaviate.WeaviateClient)
    with pytest.raises(AsyncClientNotProvidedError):
        vector_store.async_client


# Functional tests for new embedding features


@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock isinstance check
    client.__class__ = weaviate.WeaviateClient
    # Mock batch context manager
    batch_mock = MagicMock()
    client.batch.dynamic.return_value.__enter__.return_value = batch_mock
    # Ensure collections attribute exists
    client.collections = MagicMock()
    # Mock schema existence check
    client.collections.exists.return_value = True
    return client


@pytest.fixture
def mock_async_client():
    client = MagicMock()
    client.__class__ = weaviate.WeaviateAsyncClient
    client.collections = MagicMock()
    # Mock schema existence check
    client.collections.exists.return_value = True
    # Mock async methods
    client.collections.get.return_value.data.insert_many = AsyncMock()
    client.collections.get.return_value.query.hybrid = AsyncMock()
    return client


def test_add_with_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=True
    )

    node = TextNode(text="test", embedding=[0.1, 0.2])
    vector_store.add([node])

    # Verify vector is None in add_object call
    batch_mock = mock_client.batch.dynamic.return_value.__enter__.return_value
    call_args = batch_mock.add_object.call_args
    assert call_args.kwargs["vector"] is None


def test_add_without_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=False
    )

    node = TextNode(text="test", embedding=[0.1, 0.2])
    vector_store.add([node])

    # Verify vector is passed
    batch_mock = mock_client.batch.dynamic.return_value.__enter__.return_value
    call_args = batch_mock.add_object.call_args
    assert call_args.kwargs["vector"] == [0.1, 0.2]


def test_query_with_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=True
    )

    query = VectorStoreQuery(query_embedding=[0.1, 0.2], query_str="test")
    vector_store.query(query)

    # Verify vector is None in query
    collection_mock = mock_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] is None


def test_query_without_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=False
    )

    query = VectorStoreQuery(query_embedding=[0.1, 0.2], query_str="test")
    vector_store.query(query)

    # Verify vector is passed
    collection_mock = mock_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_async_add_with_native_embedding(mock_async_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_async_client, index_name="Test", native_embedding=True
    )

    # Mock schema check to avoid actual calls
    vector_store._collection_initialized = True

    node = TextNode(text="test", embedding=[0.1, 0.2])
    await vector_store.async_add([node])

    collection_mock = mock_async_client.collections.get.return_value
    call_args = collection_mock.data.insert_many.call_args
    # Check the first object in the list passed to insert_many
    inserted_objects = call_args.args[0]
    assert inserted_objects[0].vector is None


@pytest.mark.asyncio
async def test_async_add_without_native_embedding(mock_async_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_async_client,
        index_name="Test",
        native_embedding=False,
    )

    vector_store._collection_initialized = True

    node = TextNode(text="test", embedding=[0.1, 0.2])
    await vector_store.async_add([node])

    collection_mock = mock_async_client.collections.get.return_value
    call_args = collection_mock.data.insert_many.call_args
    inserted_objects = call_args.args[0]
    assert inserted_objects[0].vector == [0.1, 0.2]


@pytest.mark.asyncio
async def test_async_query_with_native_embedding(mock_async_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_async_client, index_name="Test", native_embedding=True
    )

    query = VectorStoreQuery(query_embedding=[0.1, 0.2], query_str="test")
    await vector_store.aquery(query)

    # Verify vector is None in query
    collection_mock = mock_async_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] is None


@pytest.mark.asyncio
async def test_async_query_without_native_embedding(mock_async_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_async_client,
        index_name="Test",
        native_embedding=False,
    )

    query = VectorStoreQuery(query_embedding=[0.1, 0.2], query_str="test")
    await vector_store.aquery(query)

    # Verify vector is passed
    collection_mock = mock_async_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] == [0.1, 0.2]


def test_query_hybrid_with_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=True
    )

    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        query_str="test",
        mode=VectorStoreQueryMode.HYBRID,
        alpha=0.75,
    )
    vector_store.query(query)

    # Verify vector is None and alpha is passed
    collection_mock = mock_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] is None

    assert call_args.kwargs["alpha"] == 0.75


def test_query_hybrid_without_native_embedding(mock_client):
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=False
    )

    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        query_str="test",
        mode=VectorStoreQueryMode.HYBRID,
        alpha=0.75,
    )
    vector_store.query(query)

    # Verify vector is passed and alpha is passed
    collection_mock = mock_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] == [0.1, 0.2]
    assert call_args.kwargs["alpha"] == 0.75


def test_query_with_native_embedding_no_query_str(mock_client):
    """Test that ValueError is raised when native_embedding=True but query_str is missing."""
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=True
    )

    query = VectorStoreQuery(query_embedding=[0.1, 0.2], query_str="")

    with pytest.raises(ValueError) as exc:
        vector_store.query(query)

    assert "When native_embedding=True, a non-empty query_str must be provided" in str(
        exc.value
    )


def test_query_default_with_native_embedding(mock_client):
    """Test DEFAULT query mode with native_embedding=True."""
    vector_store = WeaviateVectorStore(
        weaviate_client=mock_client, index_name="Test", native_embedding=True
    )

    query = VectorStoreQuery(
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    vector_store.query(query)

    # Verify vector is None
    collection_mock = mock_client.collections.get.return_value
    call_args = collection_mock.query.hybrid.call_args
    assert call_args.kwargs["vector"] is None
    # Verify query string is passed
    assert call_args.kwargs["query"] == "test"
