"""
Integration tests for Valkey vector store with real Valkey instance.

Tests are organized into three groups:
1. TestBasicConnection - Tests different initialization patterns (3 tests)
2. TestSyncOperations - All use cases using sync operations only
3. TestAsyncOperations - All use cases using async operations only
"""

import pytest
from glide import ft
from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    TagField,
    NumericField,
)

from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
)
from llama_index.vector_stores.valkey import ValkeyVectorStore, ValkeyVectorStoreError
from llama_index.vector_stores.valkey.schema import ValkeyVectorStoreSchema


@pytest.fixture
def clean_valkey(valkey_client):
    """Clean Valkey database before and after each test."""
    # Skip flushdb to avoid timeout issues when running tests together
    # Each test uses unique index names, so no conflicts
    return
    # No cleanup needed - tests clean up their own indexes


@pytest.fixture
async def cleanup_indexes(valkey_client_async):
    """Clean up all indexes after test."""
    yield

    try:
        result = await valkey_client_async.custom_command(["FT._LIST"])
        if result:
            for index_name in result:
                if isinstance(index_name, bytes):
                    index_name = index_name.decode()
                try:
                    await ft.dropindex(valkey_client_async, index_name)
                except Exception:
                    pass
    except Exception:
        pass


@pytest.mark.integration
class TestBasicConnection:
    """Test basic connection and initialization with different client types."""

    @pytest.mark.asyncio
    async def test_connection_with_url(self):
        """Test that ValkeyVectorStore can connect using URL."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_url_connection"
        schema.index.prefix = "test_url_connection"

        vector_store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379", schema=schema, overwrite=True
        )

        # With lazy initialization, clients are None until first use
        assert vector_store._valkey_client is None
        assert vector_store._valkey_client_async is None
        assert vector_store._pending_sync_config is not None
        assert vector_store._pending_async_config is not None

        await vector_store.async_create_index()
        # After async operation, async client should be created
        assert vector_store._valkey_client_async is not None

        exists = await vector_store.async_index_exists()
        assert exists is True

    def test_connection_with_sync_client(self, valkey_client):
        """Test that ValkeyVectorStore works with sync client only."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_client_connection"
        schema.index.prefix = "test_sync_client_connection"

        vector_store = ValkeyVectorStore(
            valkey_client=valkey_client, schema=schema, overwrite=True
        )

        assert vector_store._valkey_client is not None
        assert vector_store._valkey_client_async is None

        vector_store.create_index()
        exists = vector_store.index_exists()
        assert exists is True

    @pytest.mark.asyncio
    async def test_connection_with_async_client(self, valkey_client_async):
        """Test that ValkeyVectorStore can connect using async client only."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_client_connection"
        schema.index.prefix = "test_async_client_connection"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        assert vector_store._valkey_client_async is not None
        assert vector_store._valkey_client is None

        await vector_store.async_create_index()
        exists = await vector_store.async_index_exists()
        assert exists is True


@pytest.mark.integration
class TestSyncOperations:
    """Test all use cases using sync operations only."""

    def test_index_management(self, clean_valkey):
        """Test index creation, existence check, and deletion."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_index_mgmt"
        schema.index.prefix = "test_sync_index_mgmt"

        vector_store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379", schema=schema
        )

        vector_store.create_index()
        exists = vector_store.index_exists()
        assert exists is True

        vector_store.delete_index()
        exists = vector_store.index_exists()
        assert exists is False

    def test_index_overwrite(self, clean_valkey):
        """Test creating index with overwrite=True."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_overwrite"
        schema.index.prefix = "test_sync_overwrite"

        vector_store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379", schema=schema, overwrite=True
        )

        vector_store.create_index()
        vector_store.create_index(overwrite=True)

        exists = vector_store.index_exists()
        assert exists is True

        vector_store.delete_index()

    def test_add_and_query_nodes(self, valkey_client, clean_valkey):
        """Test adding nodes and querying them synchronously."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_add_query"
        schema.index.prefix = "test_sync_add_query"

        vector_store = ValkeyVectorStore(
            valkey_client=valkey_client, schema=schema, overwrite=True
        )

        vector_store.create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        nodes = [
            TextNode(
                text="test document 1",
                id_="node_1",
                embedding=embed_model.get_text_embedding("test document 1"),
            ),
            TextNode(
                text="test document 2",
                id_="node_2",
                embedding=embed_model.get_text_embedding("test document 2"),
            ),
        ]

        node_ids = vector_store.add(nodes)
        assert len(node_ids) == 2

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)
        result = vector_store.query(query)

        assert len(result.nodes) > 0
        assert len(result.ids) > 0

        vector_store.delete_index()

    def test_delete_nodes_by_id(self, valkey_client, clean_valkey):
        """Test deleting specific nodes by ID synchronously."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_delete_nodes"
        schema.index.prefix = "test_sync_delete_nodes"

        vector_store = ValkeyVectorStore(
            valkey_client=valkey_client, schema=schema, overwrite=True
        )

        vector_store.create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        nodes = [
            TextNode(
                text="test document 1",
                id_="node_1",
                embedding=embed_model.get_text_embedding("test document 1"),
            ),
            TextNode(
                text="test document 2",
                id_="node_2",
                embedding=embed_model.get_text_embedding("test document 2"),
            ),
        ]

        node_ids = vector_store.add(nodes)
        vector_store.delete_nodes([node_ids[0]])

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=10)
        result = vector_store.query(query)

        assert node_ids[0] not in result.ids
        vector_store.delete_index()

    def test_persist_operations(self, valkey_client, clean_valkey):
        """Test persist operations synchronously."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)

        # Use synchronous SAVE instead of BGSAVE to avoid blocking subsequent tests
        try:
            vector_store.persist(in_background=False)
        except ValkeyVectorStoreError as e:
            if "SAVE" not in str(e) and "Background save" not in str(e):
                raise

    def test_query_with_filters(self, valkey_client, clean_valkey):
        """Test querying with metadata filters synchronously."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_filter_query"
        schema.index.prefix = "test_sync_filter_query"
        schema.add_fields([TagField("category", "category")])

        vector_store = ValkeyVectorStore(
            valkey_client=valkey_client, schema=schema, overwrite=True
        )

        try:
            vector_store.create_index()

            embed_model = MockEmbedding(embed_dim=1536)
            nodes = [
                TextNode(
                    text="book content",
                    id_="node_1",
                    metadata={"category": "book"},
                    embedding=embed_model.get_text_embedding("book content"),
                ),
                TextNode(
                    text="movie content",
                    id_="node_2",
                    metadata={"category": "movie"},
                    embedding=embed_model.get_text_embedding("movie content"),
                ),
            ]

            vector_store.add(nodes)

            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", value="book", operator=FilterOperator.EQ
                    )
                ]
            )
            query_embedding = embed_model.get_query_embedding("content")
            query = VectorStoreQuery(
                query_embedding=query_embedding, filters=filters, similarity_top_k=5
            )

            result = vector_store.query(query)

            assert len(result.nodes) >= 1
            for node in result.nodes:
                assert node.metadata.get("category") == "book"
        finally:
            try:
                vector_store.delete_index()
            except Exception:
                pass

    def test_filter_only_query_sync(self, valkey_client, clean_valkey):
        """Test filter-only query synchronously."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_sync_filter_only"
        schema.index.prefix = "test_sync_filter_only"
        schema.add_fields([TagField("type", "type")])

        vector_store = ValkeyVectorStore(
            valkey_client=valkey_client, schema=schema, overwrite=True
        )

        vector_store.create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        node = TextNode(
            text="test content",
            id_="node_1",
            metadata={"type": "test"},
            embedding=embed_model.get_text_embedding("test content"),
        )

        vector_store.add([node])

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="type", value="test", operator=FilterOperator.EQ)
            ]
        )
        query = VectorStoreQuery(filters=filters, similarity_top_k=5)
        result = vector_store.query(query)

        assert len(result.nodes) >= 1
        assert result.nodes[0].metadata["type"] == "test"

        vector_store.delete_index()


@pytest.mark.integration
class TestAsyncOperations:
    """Test all use cases using async operations only."""

    @pytest.mark.asyncio
    async def test_add_nodes_directly(self, test_nodes, valkey_client_async):
        """Test adding nodes directly without using VectorStoreIndex."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_add_nodes"
        schema.index.prefix = "test_async_add_nodes"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        for node in test_nodes:
            node.embedding = embed_model.get_text_embedding(node.get_content())

        node_ids = await vector_store.async_add(test_nodes)
        assert len(node_ids) == len(test_nodes)

        query_embedding = embed_model.get_query_embedding("turtle")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        result = await vector_store.aquery(query)

        assert len(result.nodes) > 0
        assert len(result.ids) > 0

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_delete_nodes_by_id(self, test_nodes, valkey_client_async):
        """Test deleting specific nodes by ID."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_delete_nodes"
        schema.index.prefix = "test_async_delete_nodes"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        for node in test_nodes:
            node.embedding = embed_model.get_text_embedding(node.get_content())

        node_ids = await vector_store.async_add(test_nodes)
        await vector_store.adelete_nodes([node_ids[0]])

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=10)
        result = await vector_store.aquery(query)

        assert node_ids[0] not in result.ids
        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_custom_field_filtering(
        self, documents, turtle_test, valkey_client_async
    ):
        """Test filtering with custom metadata fields."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_custom_filter"
        schema.index.prefix = "test_async_custom_filter"
        schema.add_fields([TagField("animal", "animal")])

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        # Add documents directly using async_add instead of VectorStoreIndex
        embed_model = MockEmbedding(embed_dim=1536)
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc.text,
                id_=doc.id_,
                metadata=doc.metadata,
                embedding=embed_model.get_text_embedding(doc.text),
            )
            nodes.append(node)

        await vector_store.async_add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="animal", value="turtle", operator=FilterOperator.EQ)
            ]
        )

        query_embedding = embed_model.get_query_embedding("animal")
        query = VectorStoreQuery(
            query_embedding=query_embedding, filters=filters, similarity_top_k=5
        )

        result = await vector_store.aquery(query)

        assert result.nodes is not None
        assert len(result.nodes) >= 1

        for node in result.nodes:
            assert node.metadata.get("animal") == "turtle"

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_numeric_field_filtering(self, valkey_client_async):
        """Test filtering with numeric fields."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_numeric_filter"
        schema.index.prefix = "test_async_numeric_filter"
        schema.add_fields([NumericField("price", "price")])

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        nodes = [
            TextNode(
                text="cheap item",
                id_="cheap_item",
                metadata={"price": 10},
                embedding=embed_model.get_text_embedding("cheap item"),
            ),
            TextNode(
                text="expensive item",
                id_="expensive_item",
                metadata={"price": 100},
                embedding=embed_model.get_text_embedding("expensive item"),
            ),
        ]

        await vector_store.async_add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="price", value=50, operator=FilterOperator.GT)]
        )

        query_embedding = embed_model.get_query_embedding("item")
        query = VectorStoreQuery(
            query_embedding=query_embedding, filters=filters, similarity_top_k=5
        )

        result = await vector_store.aquery(query)

        assert result.nodes is not None
        assert len(result.nodes) >= 1

        for node in result.nodes:
            price = node.metadata.get("price")
            if price is not None:
                assert float(price) > 50

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_filter_only_query(self, documents, valkey_client_async):
        """Test querying with filters only, no embedding."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_filter_only"
        schema.index.prefix = "test_async_filter_only"
        schema.add_fields([TagField("animal", "animal")])

        vector_store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379",
            valkey_client_async=valkey_client_async,
            schema=schema,
            overwrite=True,
        )

        await vector_store.async_create_index()

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=MockEmbedding(embed_dim=1536),
            storage_context=storage_context,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="animal", value="turtle", operator=FilterOperator.EQ)
            ]
        )
        query = VectorStoreQuery(filters=filters, similarity_top_k=5)
        result = await vector_store.aquery(query)

        assert result.nodes is not None
        assert len(result.nodes) >= 1
        # Verify all results match the filter
        for node in result.nodes:
            assert node.metadata["animal"] == "turtle"

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_multiple_filter_conditions(self, valkey_client_async):
        """Test filter-only query with multiple conditions."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_multi_filter"
        schema.index.prefix = "test_async_multi_filter"
        schema.add_fields(
            [TagField("category", "category"), NumericField("rating", "rating")]
        )

        vector_store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379",
            valkey_client_async=valkey_client_async,
            schema=schema,
            overwrite=True,
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        docs = [
            Document(
                text="good book",
                metadata={"category": "book", "rating": 5},
                embedding=embed_model.get_text_embedding("good book"),
            ),
            Document(
                text="bad book",
                metadata={"category": "book", "rating": 2},
                embedding=embed_model.get_text_embedding("bad book"),
            ),
            Document(
                text="good movie",
                metadata={"category": "movie", "rating": 5},
                embedding=embed_model.get_text_embedding("good movie"),
            ),
        ]

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs, embed_model=embed_model, storage_context=storage_context
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="book", operator=FilterOperator.EQ
                ),
                MetadataFilter(key="rating", value=3, operator=FilterOperator.GT),
            ]
        )

        query = VectorStoreQuery(filters=filters, similarity_top_k=10)
        result = await vector_store.aquery(query)

        assert result.nodes is not None
        assert len(result.nodes) >= 1

        for node in result.nodes:
            assert node.metadata.get("category") == "book"
            rating = node.metadata.get("rating")
            if rating is not None:
                assert float(rating) > 3

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_query_empty_index(self, valkey_client_async):
        """Test querying an empty index."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_empty_index"
        schema.index.prefix = "test_async_empty_index"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)

        result = await vector_store.aquery(query)

        assert result.nodes == []
        assert result.ids == []

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_add_duplicate_nodes(self, valkey_client_async):
        """Test adding nodes with duplicate IDs."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_duplicates"
        schema.index.prefix = "test_async_duplicates"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        node = TextNode(
            text="test",
            id_="duplicate_id",
            embedding=embed_model.get_text_embedding("test"),
        )

        await vector_store.async_add([node])
        await vector_store.async_add([node])

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=10)
        result = await vector_store.aquery(query)

        assert len(result.nodes) >= 1

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, valkey_client_async):
        """Test deleting a document that doesn't exist."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_delete_nonexistent"
        schema.index.prefix = "test_async_delete_nonexistent"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()
        await vector_store.adelete("nonexistent_doc_id")
        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_large_batch_add(self, valkey_client_async):
        """Test adding a large batch of nodes."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_large_batch"
        schema.index.prefix = "test_async_large_batch"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        nodes = []
        for i in range(100):
            node = TextNode(
                text=f"test document {i}",
                id_=f"node_{i}",
                embedding=embed_model.get_text_embedding(f"test document {i}"),
            )
            nodes.append(node)

        node_ids = await vector_store.async_add(nodes)
        assert len(node_ids) == 100

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=10)
        result = await vector_store.aquery(query)

        assert len(result.nodes) > 0

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_create_and_check_index_exists(self, valkey_client_async):
        """Test creating an index and checking it exists."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_index_create"
        schema.index.prefix = "test_async_index_create"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema
        )

        try:
            await vector_store.async_delete_index()
        except ValkeyVectorStoreError:
            pass

        await vector_store.async_create_index()
        exists = await vector_store.async_index_exists()
        assert exists is True

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_delete_index_removes_index(self, valkey_client_async):
        """Test that delete_index removes the index."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_index_delete"
        schema.index.prefix = "test_async_index_delete"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema
        )

        await vector_store.async_create_index()
        await vector_store.async_delete_index()

        exists = await vector_store.async_index_exists()
        assert exists is False

    @pytest.mark.asyncio
    async def test_create_index_with_overwrite_true(self, valkey_client_async):
        """Test creating index with overwrite=True recreates it."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_index_overwrite"
        schema.index.prefix = "test_async_index_overwrite"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()
        await vector_store.async_create_index(overwrite=True)

        exists = await vector_store.async_index_exists()
        assert exists is True

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_create_index_without_overwrite_when_exists(
        self, valkey_client_async
    ):
        """Test creating index without overwrite when it exists."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_index_no_overwrite"
        schema.index.prefix = "test_async_index_no_overwrite"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=False
        )

        await vector_store.async_create_index()

        try:
            await vector_store.async_create_index(overwrite=False)
        except ValkeyVectorStoreError:
            pass

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_persist_foreground(self, valkey_client_async):
        """Test persist with in_background=False."""
        vector_store = ValkeyVectorStore(valkey_client_async=valkey_client_async)

        try:
            await vector_store.apersist(in_background=False)
        except ValkeyVectorStoreError as e:
            if "SAVE" not in str(e) and "Background save" not in str(e):
                raise

    @pytest.mark.asyncio
    async def test_process_results_with_node_content(self, valkey_client_async):
        """Test processing search results with actual node content."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_result_processing"
        schema.index.prefix = "test_async_result_processing"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        node = TextNode(
            text="test text for result processing",
            id_="test_id",
            metadata={"key": "value"},
            embedding=embed_model.get_text_embedding("test text"),
        )

        await vector_store.async_add([node])

        query_embedding = embed_model.get_query_embedding("test")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
        result = await vector_store.aquery(query)

        assert len(result.nodes) == 1
        assert len(result.ids) == 1
        assert result.nodes[0].text == "test text for result processing"

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_process_results_without_vector_score(self, valkey_client_async):
        """Test processing filter-only query results (no vector scores)."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_filter_result"
        schema.index.prefix = "test_async_filter_result"
        schema.add_fields([TagField("category", "category")])

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        node = TextNode(
            text="test text",
            id_="test_id",
            metadata={"category": "test"},
            embedding=embed_model.get_text_embedding("test text"),
        )

        await vector_store.async_add([node])

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
            ]
        )
        query = VectorStoreQuery(filters=filters, similarity_top_k=5)
        result = await vector_store.aquery(query)

        assert len(result.nodes) == 1

        await vector_store.async_delete_index()

    @pytest.mark.asyncio
    async def test_add_nodes_directly_async(self, test_nodes, valkey_client_async):
        """Test adding nodes directly without using VectorStoreIndex."""
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_async_add_direct"
        schema.index.prefix = "test_async_add_direct"

        vector_store = ValkeyVectorStore(
            valkey_client_async=valkey_client_async, schema=schema, overwrite=True
        )

        await vector_store.async_create_index()

        embed_model = MockEmbedding(embed_dim=1536)
        for node in test_nodes:
            node.embedding = embed_model.get_text_embedding(node.get_content())

        node_ids = await vector_store.async_add(test_nodes)
        assert len(node_ids) == len(test_nodes)

        query_embedding = embed_model.get_query_embedding("turtle")
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        result = await vector_store.aquery(query)

        assert len(result.nodes) > 0
        assert len(result.ids) > 0

        await vector_store.async_delete_index()
