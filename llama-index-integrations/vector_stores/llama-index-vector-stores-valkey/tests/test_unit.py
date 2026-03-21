"""Unit tests for Valkey vector store components."""

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
)
from llama_index.vector_stores.valkey import ValkeyVectorStore
from llama_index.vector_stores.valkey.schema import (
    ValkeyVectorStoreSchema,
    NODE_ID_FIELD_NAME,
    DOC_ID_FIELD_NAME,
    TEXT_FIELD_NAME,
    VECTOR_FIELD_NAME,
)


class TestClassInheritance:
    """Test class structure and inheritance."""

    def test_inherits_from_base_pydantic_vector_store(self):
        """Test that ValkeyVectorStore inherits from BasePydanticVectorStore."""
        names_of_base_classes = [b.__name__ for b in ValkeyVectorStore.__mro__]
        assert BasePydanticVectorStore.__name__ in names_of_base_classes


class TestSchemaValidation:
    """Test schema validation and configuration."""

    def test_default_schema_has_required_fields(self):
        schema = ValkeyVectorStoreSchema()
        field_names = [f.name for f in schema.fields if hasattr(f, "name")]

        assert NODE_ID_FIELD_NAME in field_names
        assert DOC_ID_FIELD_NAME in field_names
        assert TEXT_FIELD_NAME in field_names
        assert VECTOR_FIELD_NAME in field_names

    def test_custom_field_addition(self):
        from glide_shared.commands.server_modules.ft_options.ft_create_options import (
            TagField,
        )

        schema = ValkeyVectorStoreSchema()
        schema.add_fields([TagField("category", "category")])
        field_names = [f.name for f in schema.fields if hasattr(f, "name")]
        assert "category" in field_names

    def test_multiple_custom_fields(self):
        from glide_shared.commands.server_modules.ft_options.ft_create_options import (
            TagField,
            NumericField,
        )

        schema = ValkeyVectorStoreSchema()
        schema.add_fields(
            [TagField("category", "category"), NumericField("price", "price")]
        )
        field_names = [f.name for f in schema.fields if hasattr(f, "name")]
        assert "category" in field_names
        assert "price" in field_names

    def test_schema_index_configuration(self):
        schema = ValkeyVectorStoreSchema()
        assert schema.index.name == "llama_index"
        assert schema.index.prefix == "llama_index/vector"

    def test_custom_index_name(self):
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "custom_index"
        assert schema.index.name == "custom_index"


class TestInitialization:
    """Test ValkeyVectorStore initialization."""

    def test_init_with_client(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        assert vector_store.client is not None
        assert vector_store.index_name == "llama_index"

    def test_init_without_client_or_url_raises_error(self):
        with pytest.raises(
            Exception, match="Either valkey_client, valkey_url, or valkey_client_async"
        ):
            ValkeyVectorStore()

    def test_client_property_returns_sync_client(self, valkey_client):
        """Test that client property returns sync client."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        assert vector_store.client is valkey_client

    def test_client_property_raises_when_no_sync_client(self, valkey_client_async):
        """Test that client property raises error when only async client available."""
        vector_store = ValkeyVectorStore(valkey_client_async=valkey_client_async)
        with pytest.raises(Exception, match="No sync client available"):
            _ = vector_store.client

    def test_aclient_property_returns_async_client(self, valkey_client_async):
        """Test that aclient property returns async client."""
        vector_store = ValkeyVectorStore(valkey_client_async=valkey_client_async)
        assert vector_store.aclient is valkey_client_async

    def test_aclient_property_raises_when_no_async_client(self, valkey_client):
        """Test that aclient property raises error when only sync client available."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        with pytest.raises(Exception, match="No async client available"):
            _ = vector_store.aclient

    def test_property_accessors(self, valkey_client):
        schema = ValkeyVectorStoreSchema()
        schema.index.name = "test_index"
        vector_store = ValkeyVectorStore(valkey_client=valkey_client, schema=schema)

        assert vector_store.index_name == "test_index"
        assert vector_store.schema == schema
        assert vector_store.client is not None

    def test_set_return_fields(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        vector_store.set_return_fields(["field1", "field2"])


class TestEmbeddingValidation:
    """Test embedding dimension validation."""

    def test_empty_node_list_returns_empty(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        result = vector_store.add([])
        assert result == []

    def test_wrong_embedding_dimension_raises_error(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        node = TextNode(text="test", embedding=[0.0] * 512)

        with pytest.raises(
            ValueError, match="Attempting to index embeddings of dim 512"
        ):
            vector_store.add([node])

    def test_correct_embedding_dimension_accepted(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        node = TextNode(
            text="test",
            id_="test_id",
            embedding=[0.0] * 1536,
        )
        # Should not raise
        try:
            result = vector_store.add([node])
            # Result should be list of IDs
            assert isinstance(result, list)
        except Exception as e:
            if "Attempting to index embeddings" in str(e):
                pytest.fail("Embedding dimension validation failed incorrectly")


class TestURLParsing:
    """Test URL parsing for valkey_url parameter."""

    def test_url_with_port(self):
        """Test parsing URL with explicit port."""
        vector_store = ValkeyVectorStore(valkey_url="valkey://localhost:6380")
        # Verify config was stored (no connection made yet)
        assert vector_store._pending_sync_config is not None
        assert vector_store._pending_sync_config.addresses[0].host == "localhost"
        assert vector_store._pending_sync_config.addresses[0].port == 6380

    def test_url_without_port(self):
        """Test parsing URL without port (should default to 6379)."""
        vector_store = ValkeyVectorStore(valkey_url="valkey://localhost")
        assert vector_store._pending_sync_config is not None
        assert vector_store._pending_sync_config.addresses[0].host == "localhost"
        assert vector_store._pending_sync_config.addresses[0].port == 6379

    def test_url_with_ip_address(self):
        """Test parsing URL with IP address."""
        vector_store = ValkeyVectorStore(valkey_url="valkey://192.168.1.100:6379")
        assert vector_store._pending_sync_config is not None
        assert vector_store._pending_sync_config.addresses[0].host == "192.168.1.100"
        assert vector_store._pending_sync_config.addresses[0].port == 6379

    def test_url_with_domain(self):
        """Test parsing URL with domain name."""
        vector_store = ValkeyVectorStore(valkey_url="valkey://example.com:6379")
        assert vector_store._pending_sync_config is not None
        assert vector_store._pending_sync_config.addresses[0].host == "example.com"
        assert vector_store._pending_sync_config.addresses[0].port == 6379


class TestFilterProcessing:
    """Test metadata filter processing."""

    def test_build_filter_string_no_filters(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        result = vector_store._build_filter_string(None)
        assert result == "*"

    def test_build_filter_string_empty_filters(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(filters=[])
        result = vector_store._build_filter_string(filters)
        assert result == "*"

    def test_equality_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="animal", value="turtle", operator=FilterOperator.EQ)
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@animal:{turtle}" in result

    def test_not_equal_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="animal", value="turtle", operator=FilterOperator.NE)
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "-@animal:{turtle}" in result

    def test_greater_than_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[MetadataFilter(key="price", value=100, operator=FilterOperator.GT)]
        )
        result = vector_store._build_filter_string(filters)
        assert "@price:[(100 +inf]" in result

    def test_less_than_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[MetadataFilter(key="price", value=100, operator=FilterOperator.LT)]
        )
        result = vector_store._build_filter_string(filters)
        assert "@price:[-inf (100]" in result

    def test_greater_than_equal_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="price", value=100, operator=FilterOperator.GTE)
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@price:[100 +inf]" in result

    def test_less_than_equal_filter(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="price", value=100, operator=FilterOperator.LTE)
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@price:[-inf 100]" in result

    def test_multiple_filters_combined(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="animal", value="turtle", operator=FilterOperator.EQ
                ),
                MetadataFilter(key="price", value=100, operator=FilterOperator.GT),
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@animal:{turtle}" in result
        assert "@price:[(100 +inf]" in result

    def test_filter_with_space_in_value(self, valkey_client):
        """Test filter value containing spaces."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="name", value="hello world", operator=FilterOperator.EQ
                )
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@name:{hello world}" in result

    def test_filter_with_at_symbol(self, valkey_client):
        """Test filter value containing @ symbol."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="email", value="test@example.com", operator=FilterOperator.EQ
                )
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@email:{test@example.com}" in result

    def test_filter_with_dash(self, valkey_client):
        """Test filter value containing dash."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="id", value="value-with-dash", operator=FilterOperator.EQ
                )
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@id:{value-with-dash}" in result

    def test_filter_with_parentheses(self, valkey_client):
        """Test filter value containing parentheses."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="note", value="test(2024)", operator=FilterOperator.EQ
                )
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@note:{test(2024)}" in result

    def test_filter_with_comma(self, valkey_client):
        """Test filter value containing comma."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value="a,b,c", operator=FilterOperator.EQ)
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@tags:{a,b,c}" in result

    def test_filter_with_multiple_special_chars(self, valkey_client):
        """Test filter value with multiple special characters."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="email",
                    value="test@example.com (2024)",
                    operator=FilterOperator.EQ,
                )
            ]
        )
        result = vector_store._build_filter_string(filters)
        assert "@email:{test@example.com (2024)}" in result


class TestQueryValidation:
    """Test query validation."""

    def test_query_without_embedding_or_filters_raises_error(self, valkey_client):
        """Sync query raises error when no embedding or filters provided."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        query = VectorStoreQuery(similarity_top_k=5)

        with pytest.raises(
            ValueError, match="Either query_embedding or metadata filters are required"
        ):
            vector_store.query(query)

    @pytest.mark.asyncio
    async def test_async_query_without_embedding_or_filters_raises_error(
        self, valkey_client_async
    ):
        vector_store = ValkeyVectorStore(valkey_client_async=valkey_client_async)
        query = VectorStoreQuery(similarity_top_k=5)

        with pytest.raises(
            ValueError, match="Either query_embedding or metadata filters are required"
        ):
            await vector_store.aquery(query)

    def test_sync_query_returns_empty_with_warning(self, valkey_client):
        """Sync query works but may return empty results if index doesn't exist or has no data."""
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)

        # Create index to avoid errors
        vector_store.create_index(overwrite=True)

        query = VectorStoreQuery(
            query_embedding=[0.0] * 1536,
            similarity_top_k=5,
        )

        result = vector_store.query(query)
        # Should return empty results since no data was added
        assert result.nodes == []
        assert result.ids == []

        # Clean up
        vector_store.delete_index()


class TestProcessSearchResults:
    """Test search result processing."""

    def test_process_empty_results(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        result = vector_store._process_search_results([], has_vector=True)
        assert result.nodes == []
        assert result.ids == []
        assert result.similarities == []

    def test_process_zero_count_results(self, valkey_client):
        vector_store = ValkeyVectorStore(valkey_client=valkey_client)
        result = vector_store._process_search_results([0], has_vector=True)
        assert result.nodes == []
        assert result.ids == []
