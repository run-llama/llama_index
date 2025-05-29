from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.core.schema import TextNode
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
import weaviate
import weaviate.embedded
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


@pytest.mark.asyncio
class TestWeaviateAsync:
    @pytest_asyncio.fixture(scope="class")
    async def async_client(self):
        client = weaviate.use_async_with_embedded()
        await client.connect()
        yield client
        await client.close()

    # This replaces the event loop which is deprecated (discussion: https://github.com/pytest-dev/pytest-asyncio/discussions/587)
    # It was necessary to implement it this way due to pytest 7 currently always being used in the pants test performed during CI.
    # TODO Revert the commit where it was implemented like this as soon as pytest >= 8 is used in the pants tests (discussion: https://github.com/run-llama/llama_index/pull/17220#issuecomment-2532175072)
    @pytest.fixture(scope="session")
    def event_loop(self):
        loop = asyncio.get_event_loop()
        yield loop
        loop.close()

    @pytest_asyncio.fixture
    async def async_vector_store(self, async_client, event_loop):
        vector_store = WeaviateVectorStore(
            weaviate_client=async_client, index_name=TEST_COLLECTION_NAME
        )
        # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
        await vector_store.aclear()
        yield vector_store
        await vector_store.aclear()

    async def test_async_basic_flow(self, async_vector_store):
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

    async def test_async_old_data_gone(self, async_vector_store):
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

    async def test_async_delete_nodes(self, async_vector_store):
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

    async def test_async_delete(self, async_vector_store):
        node_to_be_deleted = TextNode(
            text="Hello world.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_deleted")
            },
            embedding=[0.0, 0.0, 0.3],
        )
        node_to_keep = TextNode(
            text="This is a test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_kept")
            },
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

    def test_async_client_properties(self, async_vector_store):
        assert isinstance(async_vector_store.async_client, weaviate.WeaviateAsyncClient)
        with pytest.raises(SyncClientNotProvidedError):
            async_vector_store.client


class TestWeaviateSync:
    def test_class(self):
        names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
        assert BasePydanticVectorStore.__name__ in names_of_base_classes

    @pytest.fixture(scope="class")
    def client(self):
        client = weaviate.connect_to_embedded()
        yield client
        client.close()

    @pytest.fixture()
    def vector_store(self, client):
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=TEST_COLLECTION_NAME
        )
        vector_store.clear()  # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
        yield vector_store
        vector_store.clear()

    @pytest.fixture()
    def vector_store_with_sample_nodes(self, vector_store):
        nodes = [
            TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
            TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
        ]
        vector_store.add(nodes)
        return vector_store

    def test_vector_store_with_custom_batch(self, client):
        nodes = [
            TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
            TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
        ]
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
            client.batch._consistency_level
            == weaviate.classes.config.ConsistencyLevel.ONE
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
            AssertionError()
        except ValueError:
            assert True

    def test_sync_basic_flow(self, vector_store_with_sample_nodes):
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

    def test_hybrid_search(self, vector_store_with_sample_nodes):
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

    def test_query_kwargs(self, vector_store_with_sample_nodes):
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

    def test_can_query_collection_with_complex_property_types(self, client):
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

    def test_sync_delete(self, vector_store):
        node_to_be_deleted = TextNode(
            text="Hello world.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_deleted")
            },
            embedding=[0.0, 0.0, 0.3],
        )
        node_to_keep = TextNode(
            text="This is a test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_kept")
            },
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

    async def test_async_methods_called_without_async_client(self, vector_store):
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
            results = await vector_store.aquery(query)

    def test_sync_client_properties(self, vector_store):
        assert isinstance(vector_store.client, weaviate.WeaviateClient)
        with pytest.raises(AsyncClientNotProvidedError):
            vector_store.async_client
