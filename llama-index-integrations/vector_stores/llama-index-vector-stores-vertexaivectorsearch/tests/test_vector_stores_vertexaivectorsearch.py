"""Test Vertex AI Vector Store Vector Search functionality."""

import os
import uuid
import hashlib

from typing import List
from unittest.mock import patch, MagicMock

import pytest

from llama_index.core.schema import MetadataMode, TextNode, Document
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
    VectorSearchSDKManager,
)

from llama_index.vector_stores.vertexaivectorsearch import utils

from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
REGION = os.getenv("REGION", "")
INDEX_ID = os.getenv("INDEX_ID", "")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")


def set_all_env_vars() -> bool:
    """Check if all required environment variables are set."""
    return all([PROJECT_ID, REGION, INDEX_ID, ENDPOINT_ID])


def create_uuid(text: str):
    hex_string = hashlib.md5(text.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    record_data = [
        {
            "description": "A versatile pair of dark-wash denim jeans."
            "Made from durable cotton with a classic straight-leg cut, these jeans"
            " transition easily from casual days to dressier occasions.",
            "price": 65.00,
            "color": "blue",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A lightweight linen button-down shirt in a crisp white."
            " Perfect for keeping cool with breathable fabric and a relaxed fit.",
            "price": 34.99,
            "color": "white",
            "season": ["summer", "spring"],
        },
        {
            "description": "A soft, chunky knit sweater in a vibrant forest green. "
            "The oversized fit and cozy wool blend make this ideal for staying warm "
            "when the temperature drops.",
            "price": 89.99,
            "color": "green",
            "season": ["fall", "winter"],
        },
        {
            "description": "A classic crewneck t-shirt in a soft, heathered blue. "
            "Made from comfortable cotton jersey, this t-shirt is a wardrobe essential "
            "that works for every season.",
            "price": 19.99,
            "color": "blue",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A flowing midi-skirt in a delicate floral print. "
            "Lightweight and airy, this skirt adds a touch of feminine style "
            "to warmer days.",
            "price": 45.00,
            "color": "white",
            "season": ["spring", "summer"],
        },
        {
            "description": "A pair of tailored black trousers in a comfortable stretch "
            "fabric. Perfect for work or dressier events, these trousers provide a"
            " sleek, polished look.",
            "price": 59.99,
            "color": "black",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A cozy fleece hoodie in a neutral heather grey.  "
            "This relaxed sweatshirt is perfect for casual days or layering when the "
            "weather turns chilly.",
            "price": 39.99,
            "color": "grey",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A bright yellow raincoat with a playful polka dot pattern. "
            "This waterproof jacket will keep you dry and add a touch of cheer to "
            "rainy days.",
            "price": 75.00,
            "color": "yellow",
            "season": ["spring", "fall"],
        },
        {
            "description": "A pair of comfortable khaki chino shorts. These versatile "
            "shorts are a summer staple, perfect for outdoor adventures or relaxed"
            " weekends.",
            "price": 34.99,
            "color": "khaki",
            "season": ["summer"],
        },
        {
            "description": "A bold red cocktail dress with a flattering A-line "
            "silhouette. This statement piece is made from a luxurious satin fabric, "
            "ensuring a head-turning look.",
            "price": 125.00,
            "color": "red",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of classic white sneakers crafted from smooth "
            "leather. These timeless shoes offer a clean and polished look, perfect "
            "for everyday wear.",
            "price": 79.99,
            "color": "white",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A chunky cable-knit scarf in a rich burgundy color. "
            "Made from a soft wool blend, this scarf will provide warmth and a touch "
            "of classic style to cold-weather looks.",
            "price": 45.00,
            "color": "burgundy",
            "season": ["fall", "winter"],
        },
        {
            "description": "A lightweight puffer vest in a vibrant teal hue. "
            "This versatile piece adds a layer of warmth without bulk, transitioning"
            " perfectly between seasons.",
            "price": 65.00,
            "color": "teal",
            "season": ["fall", "spring"],
        },
        {
            "description": "A pair of high-waisted leggings in a sleek black."
            " Crafted from a moisture-wicking fabric with plenty of stretch, "
            "these leggings are perfect for workouts or comfortable athleisure style.",
            "price": 49.99,
            "color": "black",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A denim jacket with a faded wash and distressed details. "
            "This wardrobe staple adds a touch of effortless cool to any outfit.",
            "price": 79.99,
            "color": "blue",
            "season": ["fall", "spring", "summer"],
        },
        {
            "description": "A woven straw sunhat with a wide brim. This stylish "
            "accessory provides protection from the sun while adding a touch of "
            "summery elegance.",
            "price": 32.00,
            "color": "beige",
            "season": ["summer"],
        },
        {
            "description": "A graphic tee featuring a vintage band logo. "
            "Made from a soft cotton blend, this casual tee adds a touch of "
            "personal style to everyday looks.",
            "price": 24.99,
            "color": "white",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of well-tailored dress pants in a neutral grey. "
            "Made from a wrinkle-resistant blend, these pants look sharp and "
            "professional for workwear or formal occasions.",
            "price": 69.99,
            "color": "grey",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of classic leather ankle boots in a rich brown hue."
            " Featuring a subtle stacked heel and sleek design, these boots are perfect"
            " for elevating outfits in cooler seasons.",
            "price": 120.00,
            "color": "brown",
            "season": ["fall", "winter", "spring"],
        },
    ]

    embed_model = VertexTextEmbedding(project=PROJECT_ID, location=REGION)

    nodes = []
    for record in record_data:
        record = record.copy()
        page_content = record.pop("description")
        node_id = create_uuid(page_content)
        embedding = embed_model.get_text_embedding(page_content)
        if isinstance(page_content, str):
            metadata = {**record}
            node = TextNode(
                id_=node_id, text=page_content, embedding=embedding, metadata=metadata
            )
            nodes.append(node)
    return nodes


@pytest.mark.skipif(
    not set_all_env_vars(),
    reason="missing Vertex AI Vector Search environment variables",
)
class TestVertexAIVectorStore:
    def sdk_manager(self) -> VectorSearchSDKManager:
        return VectorSearchSDKManager(project_id=PROJECT_ID, region=REGION)

    def vector_store(self) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id=PROJECT_ID,
            region=REGION,
            index_id=INDEX_ID,
            endpoint_id=ENDPOINT_ID,
            gcs_bucket_name=GCS_BUCKET_NAME,
        )

    def test_vector_search_sdk_manager(self):
        sdk_manager = self.sdk_manager()

        if GCS_BUCKET_NAME:
            gcs_client = sdk_manager.get_gcs_client()
            assert isinstance(gcs_client, storage.Client)
            gcs_bucket = sdk_manager.get_gcs_bucket(GCS_BUCKET_NAME)
            assert isinstance(gcs_bucket, storage.Bucket)

        index = sdk_manager.get_index(index_id=INDEX_ID)
        assert isinstance(index, MatchingEngineIndex)

        endpoint = sdk_manager.get_endpoint(endpoint_id=ENDPOINT_ID)
        assert isinstance(endpoint, MatchingEngineIndexEndpoint)

    def test_add_documents(self, node_embeddings: List[TextNode]) -> None:
        """Test adding documents to Vertex AI Vector Search vector store."""
        vector_store = self.vector_store()

        # Add nodes to the Vertex AI Vector Search index
        input_doc_ids = [node_embedding.id_ for node_embedding in node_embeddings]
        doc_ids = vector_store.add(node_embeddings)

        # Ensure that all nodes are returned & they are the same as input
        assert len(doc_ids) == len(node_embeddings)
        for doc_id in doc_ids:
            assert doc_id in input_doc_ids

    def test_search(self, node_embeddings: List[TextNode]) -> None:
        """Test end to end Vertex AI Vector Search."""
        # Add nodes to the Vertex AI Vector Search index
        vector_store = self.vector_store()
        vector_store.add(node_embeddings)

        # similarity search
        embed_model = VertexTextEmbedding(project=PROJECT_ID, location=REGION)
        query = "dark-wash denim jeans"
        query_embedding = embed_model.get_query_embedding(query)
        q = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
        )
        assert result.similarities is not None

    def test_search_with_filter(self, node_embeddings: List[TextNode]) -> None:
        """Test end to end Vertex AI Vector Search with filter."""
        # Add nodes to the Vertex AI Vector Search index
        vector_store = self.vector_store()
        vector_store.add(node_embeddings)

        # similarity search
        embed_model = VertexTextEmbedding(project=PROJECT_ID, location=REGION)
        query = "I want some pants."
        query_embedding = embed_model.get_query_embedding(query)
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="color", value="blue")]
            ),
        )

        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1
        assert result.nodes[0].metadata.get("color") == "blue"

    def test_delete_doc(self) -> None:
        """Test delete document from Vertex AI Vector Search index."""
        embed_model = VertexTextEmbedding(project=PROJECT_ID, location=REGION)
        Settings.embed_model = embed_model
        vector_store = self.vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Add nodes to the Vertex AI Vector Search index
        page_content = (
            "A vibrant swimsuit with a bold geometric pattern. This fun "
            "and eye-catching piece is perfect for making a splash by the pool or at "
            "the beach."
        )
        VectorStoreIndex.from_documents(
            [
                Document(
                    doc_id=create_uuid(page_content),
                    text=page_content,
                    metadata={
                        "color": "multicolor",
                        "price": 55.00,
                        "season": ["summer"],
                    },
                ),
            ],
            storage_context=storage_context,
        )

        # similarity search
        query = "swimsuit with a bold geometric pattern"
        query_embedding = embed_model.get_query_embedding(query)
        q = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)

        result = vector_store.query(q)
        assert result.nodes is not None and len(result.nodes) == 1

        # Identify the document to delete
        ref_id_to_delete = result.nodes[0].ref_doc_id

        # Delete the document
        vector_store.delete(ref_doc_id=ref_id_to_delete)

        # Ensure that no results are returned
        result = utils.get_datapoints_by_filter(
            index=vector_store.index,
            endpoint=vector_store.endpoint,
            metadata={"ref_doc_id": ref_id_to_delete},
        )
        assert len(result) == 0

    def test_batch_update_index(self, node_embeddings: List[TextNode]) -> None:
        """Test batch update path consistency and end-to-end functionality."""
        if not GCS_BUCKET_NAME:
            pytest.skip("GCS_BUCKET_NAME not set, skipping batch update test")

        vector_store = self.vector_store()
        staging_bucket = vector_store.staging_bucket

        with patch.object(
            staging_bucket, "blob", wraps=staging_bucket.blob
        ) as mock_blob:
            initial_nodes = node_embeddings[:5]
            doc_ids = vector_store.add(initial_nodes)

            assert len(doc_ids) == len(initial_nodes)

            for call in mock_blob.call_args_list:
                path = call[0][0]
                assert path.startswith("index/")
                assert path.endswith("documents.json")

            embed_model = VertexTextEmbedding(project=PROJECT_ID, location=REGION)
            query_embedding = embed_model.get_query_embedding("denim jeans")
            q = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
            result = vector_store.query(q)

            assert result.nodes is not None and len(result.nodes) == 1
            assert result.nodes[0].id_ in doc_ids


def test_batch_update_index_path_validation():
    """Test that batch_update_index uses correct GCS path"""
    mock_bucket = MagicMock(spec=storage.Bucket)
    mock_bucket.name = "test-bucket"
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_index = MagicMock(spec=MatchingEngineIndex)
    mock_index.update_embeddings = MagicMock()

    # Create real data points using to_data_points
    ids = ["id1", "id2", "id3"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    metadatas = [
        {"field1": "value1", "field2": 10},
        {"field1": "value2", "field2": 20},
        {"field1": "value3", "field2": 30},
    ]

    data_points = utils.to_data_points(ids, embeddings, metadatas)

    utils.batch_update_index(
        index=mock_index,
        data_points=data_points,
        staging_bucket=mock_bucket,
        is_complete_overwrite=False,
    )

    assert mock_bucket.blob.called, "bucket.blob() should be called"
    blob_path = mock_bucket.blob.call_args[0][0]

    assert blob_path.startswith("index/"), (
        f"Upload path must start with 'index/' to match contents_delta_uri. Got: {blob_path}"
    )
    assert blob_path.endswith("documents.json"), (
        f"Upload path must end with 'documents.json'. Got: {blob_path}"
    )

    assert mock_index.update_embeddings.called, (
        "index.update_embeddings() should be called"
    )
    contents_delta_uri = mock_index.update_embeddings.call_args[1]["contents_delta_uri"]
    assert "index/" in contents_delta_uri, (
        f"contents_delta_uri must contain 'index/'. Got: {contents_delta_uri}"
    )


def test_class():
    names_of_base_classes = [b.__name__ for b in VertexAIVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


# =============================================================================
# V2 Unit Tests
# =============================================================================


class TestV2ParameterValidation:
    """Test parameter validation for v1 vs v2 API versions."""

    def test_v1_requires_index_id(self):
        """Test that v1 raises error when index_id is missing."""
        with pytest.raises(ValueError, match="index_id is required for v1"):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                endpoint_id="projects/test/locations/us-central1/indexEndpoints/123",
            )

    def test_v1_requires_endpoint_id(self):
        """Test that v1 raises error when endpoint_id is missing."""
        with pytest.raises(ValueError, match="endpoint_id is required for v1"):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                index_id="projects/test/locations/us-central1/indexes/123",
            )

    def test_v1_rejects_collection_id(self):
        """Test that v1 raises error when collection_id is provided."""
        with pytest.raises(
            ValueError, match="collection_id.*only valid for api_version='v2'"
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                index_id="projects/test/locations/us-central1/indexes/123",
                endpoint_id="projects/test/locations/us-central1/indexEndpoints/123",
                collection_id="my-collection",
            )

    def test_v2_requires_collection_id(self):
        """Test that v2 raises error when collection_id is missing."""
        with pytest.raises(ValueError, match="collection_id is required for v2"):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
            )

    def test_v2_rejects_index_id(self):
        """Test that v2 raises error when index_id is provided."""
        with pytest.raises(
            ValueError, match="index_id.*only valid for api_version='v1'"
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                index_id="projects/test/locations/us-central1/indexes/123",
            )

    def test_v2_rejects_endpoint_id(self):
        """Test that v2 raises error when endpoint_id is provided."""
        with pytest.raises(
            ValueError, match="endpoint_id.*only valid for api_version='v1'"
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                endpoint_id="projects/test/locations/us-central1/indexEndpoints/123",
            )

    def test_v2_rejects_gcs_bucket_name(self):
        """Test that v2 raises error when gcs_bucket_name is provided."""
        with pytest.raises(
            ValueError, match="gcs_bucket_name.*only valid for api_version='v1'"
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                gcs_bucket_name="my-bucket",
            )


class TestV2Routing:
    """Test that operations route correctly to v1 or v2 implementations."""

    @pytest.fixture
    def mock_v2_store(self):
        """Create a v2 store with mocked SDK."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VertexAIVectorStore._validate_parameters"
        ):
            return VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
            )

    def test_add_routes_to_v2(self, mock_v2_store):
        """Test that add() routes to _add_v2 when api_version='v2'."""
        with patch.object(mock_v2_store, "_add_v2", return_value=["id1"]) as mock_add:
            mock_node = MagicMock()
            mock_node.node_id = "id1"
            mock_node.get_embedding.return_value = [0.1, 0.2, 0.3]

            result = mock_v2_store.add([mock_node])

            mock_add.assert_called_once()
            assert result == ["id1"]

    def test_query_routes_to_v2(self, mock_v2_store):
        """Test that query() routes to _query_v2 when api_version='v2'."""
        mock_result = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        with patch.object(
            mock_v2_store, "_query_v2", return_value=mock_result
        ) as mock_query:
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5
            )

            result = mock_v2_store.query(query)

            mock_query.assert_called_once()
            assert result == mock_result

    def test_delete_routes_to_v2(self, mock_v2_store):
        """Test that delete() routes to _delete_v2 when api_version='v2'."""
        with patch.object(
            mock_v2_store, "_delete_v2", return_value=None
        ) as mock_delete:
            mock_v2_store.delete(ref_doc_id="doc123")

            mock_delete.assert_called_once_with("doc123")

    def test_delete_nodes_routes_to_v2(self, mock_v2_store):
        """Test that delete_nodes() routes to _delete_nodes_v2 when api_version='v2'."""
        with patch.object(
            mock_v2_store, "_delete_nodes_v2", return_value=None
        ) as mock_delete:
            mock_v2_store.delete_nodes(node_ids=["node1", "node2"])

            mock_delete.assert_called_once()

    def test_clear_routes_to_v2(self, mock_v2_store):
        """Test that clear() routes to _clear_v2 when api_version='v2'."""
        with patch.object(mock_v2_store, "_clear_v2", return_value=None) as mock_clear:
            mock_v2_store.clear()

            mock_clear.assert_called_once()


class TestV2SDKImport:
    """Test v2 SDK import error handling."""

    def test_import_v2_sdk_success(self):
        """Test that _import_v2_sdk returns the module when available."""
        # This test will pass if google-cloud-vectorsearch is installed
        try:
            from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
                _import_v2_sdk,
            )

            result = _import_v2_sdk()
            assert result is not None
            assert hasattr(result, "BatchCreateDataObjectsRequest")
        except ImportError:
            pytest.skip("google-cloud-vectorsearch not installed")

    def test_import_v2_sdk_error_message_format(self):
        """Test that ImportError message mentions the required package."""
        # This test verifies the error message format without mocking imports
        # The actual import behavior is tested by test_import_v2_sdk_success
        expected_message = "v2 operations require google-cloud-vectorsearch"

        # Verify the error message is properly formatted in the code
        import inspect
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _import_v2_sdk,
        )

        source = inspect.getsource(_import_v2_sdk)
        assert expected_message in source, (
            f"Expected error message '{expected_message}' not found in _import_v2_sdk"
        )


class TestV2FeatureFlags:
    """Test feature flag behavior for v2."""

    def test_should_use_v2_with_v2_enabled(self):
        """Test that should_use_v2 returns True when api_version='v2' and flag enabled."""
        from llama_index.vector_stores.vertexaivectorsearch.base import FeatureFlags

        with patch.object(FeatureFlags, "ENABLE_V2", True):
            assert FeatureFlags.should_use_v2("v2") is True

    def test_should_use_v2_with_v2_disabled(self):
        """Test that should_use_v2 returns False when flag is disabled."""
        from llama_index.vector_stores.vertexaivectorsearch.base import FeatureFlags

        with patch.object(FeatureFlags, "ENABLE_V2", False):
            assert FeatureFlags.should_use_v2("v2") is False

    def test_should_use_v2_with_v1_version(self):
        """Test that should_use_v2 returns False when api_version='v1'."""
        from llama_index.vector_stores.vertexaivectorsearch.base import FeatureFlags

        with patch.object(FeatureFlags, "ENABLE_V2", True):
            assert FeatureFlags.should_use_v2("v1") is False


class TestV2SDKManager:
    """Test SDK manager v2 client functionality."""

    def test_is_v2_available_when_installed(self):
        """Test is_v2_available returns True when SDK is installed."""
        try:
            import google.cloud.vectorsearch_v1beta  # noqa: F401

            sdk_manager = VectorSearchSDKManager(
                project_id="test-project",
                region="us-central1",
            )
            assert sdk_manager.is_v2_available() is True
        except ImportError:
            pytest.skip("google-cloud-vectorsearch not installed")

    def test_is_v2_available_caches_result(self):
        """Test that is_v2_available caches the result."""
        sdk_manager = VectorSearchSDKManager(
            project_id="test-project",
            region="us-central1",
        )

        # First call
        result1 = sdk_manager.is_v2_available()
        # Second call should use cached value
        result2 = sdk_manager.is_v2_available()

        assert result1 == result2
        # Check that _v2_available is set (not None)
        assert sdk_manager._v2_available is not None

    def test_get_v2_client_returns_three_clients(self):
        """Test that get_v2_client returns dict with three client types."""
        try:
            import google.cloud.vectorsearch_v1beta  # noqa: F401

            sdk_manager = VectorSearchSDKManager(
                project_id="test-project",
                region="us-central1",
            )

            clients = sdk_manager.get_v2_client()

            assert isinstance(clients, dict)
            assert "vector_search_service_client" in clients
            assert "data_object_service_client" in clients
            assert "data_object_search_service_client" in clients
        except ImportError:
            pytest.skip("google-cloud-vectorsearch not installed")


class TestV2RetryDecorator:
    """Test the retry decorator for v2 operations."""

    def test_retry_succeeds_on_first_attempt(self):
        """Test that function returns immediately on success."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import retry

        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def succeeding_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeding_function()

        assert result == "success"
        assert call_count == 1

    def test_retry_retries_on_failure(self):
        """Test that function retries on failure."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import retry

        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = failing_then_succeeding()

        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max_attempts(self):
        """Test that function raises exception after max attempts."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import retry

        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")

        with pytest.raises(Exception, match="Permanent failure"):
            always_failing()

        assert call_count == 3


# =============================================================================
# V2 Hybrid Search Unit Tests
# =============================================================================


class TestV2HybridSearchParameters:
    """Test hybrid search constructor parameter validation."""

    def test_enable_hybrid_requires_v2(self):
        """Test that enable_hybrid=True raises error for v1."""
        with pytest.raises(
            ValueError,
            match="enable_hybrid=True is only supported for api_version='v2'",
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                index_id="projects/test/locations/us-central1/indexes/123",
                endpoint_id="projects/test/locations/us-central1/indexEndpoints/123",
                enable_hybrid=True,
            )

    def test_alpha_must_be_between_0_and_1(self):
        """Test that default_hybrid_alpha must be in [0, 1] range."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                default_hybrid_alpha=1.5,
            )

    def test_alpha_negative_raises_error(self):
        """Test that negative alpha raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                default_hybrid_alpha=-0.1,
            )

    def test_hybrid_ranker_must_be_rrf_or_vertex(self):
        """Test that hybrid_ranker must be 'rrf' or 'vertex'."""
        from pydantic import ValidationError

        # Pydantic validates the Literal type
        with pytest.raises(ValidationError):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                hybrid_ranker="invalid",
            )

    def test_v2_accepts_all_hybrid_parameters(self):
        """Test that v2 accepts all hybrid parameters without error."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            store = VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                enable_hybrid=True,
                text_search_fields=["title", "content"],
                embedding_field="embedding",
                hybrid_ranker="rrf",
                default_hybrid_alpha=0.7,
                semantic_task_type="RETRIEVAL_QUERY",
                vertex_ranker_model="semantic-ranker-default@latest",
                vertex_ranker_title_field="title",
                vertex_ranker_content_field="content",
            )

            assert store.enable_hybrid is True
            assert store.text_search_fields == ["title", "content"]
            assert store.embedding_field == "embedding"
            assert store.hybrid_ranker == "rrf"
            assert store.default_hybrid_alpha == 0.7
            assert store.semantic_task_type == "RETRIEVAL_QUERY"
            assert store.vertex_ranker_model == "semantic-ranker-default@latest"
            assert store.vertex_ranker_title_field == "title"
            assert store.vertex_ranker_content_field == "content"

    def test_vertex_ranker_warning_without_fields(self):
        """Test that VertexRanker logs warning when no title/content fields configured."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch.base._logger"
            ) as mock_logger:
                VertexAIVectorStore(
                    project_id="test-project",
                    region="us-central1",
                    api_version="v2",
                    collection_id="my-collection",
                    hybrid_ranker="vertex",
                )

                mock_logger.warning.assert_called_once()
                assert "VertexRanker works best" in mock_logger.warning.call_args[0][0]


class TestV2RRFWeightCalculation:
    """Test RRF weight calculation from alpha."""

    def test_alpha_0_pure_text(self):
        """Test that alpha=0 gives pure text weight."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _calculate_rrf_weights,
        )

        weights = _calculate_rrf_weights(alpha=0.0, num_searches=2)
        assert weights == [0.0, 1.0]  # [vector, text]

    def test_alpha_1_pure_vector(self):
        """Test that alpha=1 gives pure vector weight."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _calculate_rrf_weights,
        )

        weights = _calculate_rrf_weights(alpha=1.0, num_searches=2)
        assert weights == [1.0, 0.0]  # [vector, text]

    def test_alpha_0_5_balanced(self):
        """Test that alpha=0.5 gives balanced weights."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _calculate_rrf_weights,
        )

        weights = _calculate_rrf_weights(alpha=0.5, num_searches=2)
        assert weights == [0.5, 0.5]  # [vector, text]

    def test_alpha_0_7_favors_vector(self):
        """Test that alpha=0.7 favors vector search."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _calculate_rrf_weights,
        )

        weights = _calculate_rrf_weights(alpha=0.7, num_searches=2)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(0.3)

    def test_three_searches_equal_weights(self):
        """Test that 3 searches get equal weights when not two."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _calculate_rrf_weights,
        )

        weights = _calculate_rrf_weights(alpha=0.5, num_searches=3)
        expected = 1.0 / 3
        assert all(w == pytest.approx(expected) for w in weights)


class TestV2FilterConversion:
    """Test LlamaIndex filter to V2 filter conversion."""

    def test_simple_eq_filter(self):
        """Test simple equality filter conversion."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import (
            MetadataFilters,
            MetadataFilter,
            FilterOperator,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            ]
        )

        result = _convert_filters_to_v2(filters)

        assert result == {"category": {"$eq": "tech"}}

    def test_gt_filter(self):
        """Test greater than filter conversion."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import (
            MetadataFilters,
            MetadataFilter,
            FilterOperator,
        )

        filters = MetadataFilters(
            filters=[MetadataFilter(key="price", value=50, operator=FilterOperator.GT)]
        )

        result = _convert_filters_to_v2(filters)

        assert result == {"price": {"$gt": 50}}

    def test_and_filter(self):
        """Test AND filter conversion."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import (
            MetadataFilters,
            MetadataFilter,
            FilterOperator,
            FilterCondition,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="tech", operator=FilterOperator.EQ
                ),
                MetadataFilter(key="price", value=50, operator=FilterOperator.LT),
            ],
            condition=FilterCondition.AND,
        )

        result = _convert_filters_to_v2(filters)

        assert result == {
            "$and": [
                {"category": {"$eq": "tech"}},
                {"price": {"$lt": 50}},
            ]
        }

    def test_or_filter(self):
        """Test OR filter conversion."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import (
            MetadataFilters,
            MetadataFilter,
            FilterOperator,
            FilterCondition,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="color", value="red", operator=FilterOperator.EQ),
                MetadataFilter(key="color", value="blue", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )

        result = _convert_filters_to_v2(filters)

        assert result == {
            "$or": [
                {"color": {"$eq": "red"}},
                {"color": {"$eq": "blue"}},
            ]
        }

    def test_in_filter(self):
        """Test IN filter conversion."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import (
            MetadataFilters,
            MetadataFilter,
            FilterOperator,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tags", value=["ml", "ai"], operator=FilterOperator.IN
                )
            ]
        )

        result = _convert_filters_to_v2(filters)

        assert result == {"tags": {"$in": ["ml", "ai"]}}

    def test_none_filters(self):
        """Test that None filters returns None."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )

        result = _convert_filters_to_v2(None)
        assert result is None

    def test_empty_filters(self):
        """Test that empty filters returns None."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _convert_filters_to_v2,
        )
        from llama_index.core.vector_stores.types import MetadataFilters

        result = _convert_filters_to_v2(MetadataFilters(filters=[]))
        assert result is None


class TestV2HybridQueryModes:
    """Test query mode routing and behavior."""

    @pytest.fixture
    def mock_v2_store(self):
        """Create a v2 store with hybrid enabled."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            return VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                enable_hybrid=True,
                text_search_fields=["title", "content"],
            )

    @pytest.fixture
    def mock_v2_store_no_hybrid(self):
        """Create a v2 store without hybrid enabled."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            return VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                enable_hybrid=False,
            )

    def test_hybrid_mode_requires_enable_hybrid(self, mock_v2_store_no_hybrid):
        """Test that HYBRID mode raises error without enable_hybrid=True."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_embedding=[0.1] * 768,
                    query_str="test",
                    mode=VectorStoreQueryMode.HYBRID,
                )

                with pytest.raises(ValueError, match="enable_hybrid=True"):
                    mock_v2_store_no_hybrid.query(query)

    def test_hybrid_mode_requires_query_embedding(self, mock_v2_store):
        """Test that HYBRID mode raises error without query_embedding."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_embedding=None,
                    query_str="test",
                    mode=VectorStoreQueryMode.HYBRID,
                )

                with pytest.raises(ValueError, match="query_embedding"):
                    mock_v2_store.query(query)

    def test_text_search_requires_query_str(self, mock_v2_store):
        """Test that TEXT_SEARCH mode raises error without query_str."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_str=None,
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                )

                with pytest.raises(ValueError, match="query_str"):
                    mock_v2_store.query(query)

    def test_text_search_requires_text_fields(self, mock_v2_store_no_hybrid):
        """Test that TEXT_SEARCH mode raises error without text_search_fields."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_str="test query",
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                )

                with pytest.raises(ValueError, match="text_search_fields"):
                    mock_v2_store_no_hybrid.query(query)

    def test_semantic_hybrid_requires_query_str(self, mock_v2_store):
        """Test that SEMANTIC_HYBRID mode raises error without query_str."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_embedding=[0.1] * 768,
                    query_str=None,
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                )

                with pytest.raises(ValueError, match="query_str"):
                    mock_v2_store.query(query)

    def test_sparse_raises_not_implemented(self, mock_v2_store):
        """Test that SPARSE mode raises NotImplementedError."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk"
        ):
            with patch(
                "llama_index.vector_stores.vertexaivectorsearch._sdk_manager.VectorSearchSDKManager"
            ):
                query = VectorStoreQuery(
                    query_embedding=[0.1] * 768,
                    mode=VectorStoreQueryMode.SPARSE,
                )

                with pytest.raises(
                    NotImplementedError, match="SPARSE mode is planned for Phase 2"
                ):
                    mock_v2_store.query(query)


class TestV2RankerConfiguration:
    """Test RRF vs VertexRanker configuration."""

    def test_rrf_ranker_default(self):
        """Test that RRF ranker is built by default."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            store = VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                hybrid_ranker="rrf",
            )

            assert store.hybrid_ranker == "rrf"

    def test_vertex_ranker_configuration(self):
        """Test VertexRanker configuration parameters."""
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            store = VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                hybrid_ranker="vertex",
                vertex_ranker_model="semantic-ranker-default@latest",
                vertex_ranker_title_field="title",
                vertex_ranker_content_field="content",
            )

            assert store.hybrid_ranker == "vertex"
            assert store.vertex_ranker_model == "semantic-ranker-default@latest"
            assert store.vertex_ranker_title_field == "title"
            assert store.vertex_ranker_content_field == "content"

    def test_build_ranker_rrf(self):
        """Test _build_ranker creates RRF ranker."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _build_ranker,
        )
        from llama_index.core.vector_stores.types import VectorStoreQuery

        mock_vectorsearch = MagicMock()
        mock_ranker = MagicMock()
        mock_rrf = MagicMock()
        mock_vectorsearch.Ranker.return_value = mock_ranker
        mock_vectorsearch.ReciprocalRankFusion.return_value = mock_rrf

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk",
            return_value=mock_vectorsearch,
        ):
            mock_store = MagicMock()
            mock_store.hybrid_ranker = "rrf"
            mock_store.default_hybrid_alpha = 0.5

            query = VectorStoreQuery(query_embedding=[0.1] * 768, alpha=0.5)

            result = _build_ranker(mock_store, query, num_searches=2)

            mock_vectorsearch.ReciprocalRankFusion.assert_called_once_with(
                weights=[0.5, 0.5]
            )
            mock_vectorsearch.Ranker.assert_called_once()
            assert result == mock_ranker

    def test_build_ranker_vertex(self):
        """Test _build_ranker creates VertexRanker."""
        from llama_index.vector_stores.vertexaivectorsearch._v2_operations import (
            _build_ranker,
        )
        from llama_index.core.vector_stores.types import VectorStoreQuery

        mock_vectorsearch = MagicMock()
        mock_ranker = MagicMock()
        mock_vertex_ranker = MagicMock()
        mock_vectorsearch.Ranker.return_value = mock_ranker
        mock_vectorsearch.VertexRanker.return_value = mock_vertex_ranker

        with patch(
            "llama_index.vector_stores.vertexaivectorsearch._v2_operations._import_v2_sdk",
            return_value=mock_vectorsearch,
        ):
            mock_store = MagicMock()
            mock_store.hybrid_ranker = "vertex"
            mock_store.vertex_ranker_model = "semantic-ranker-default@latest"
            mock_store.vertex_ranker_title_field = "title"
            mock_store.vertex_ranker_content_field = "content"

            query = VectorStoreQuery(
                query_embedding=[0.1] * 768,
                query_str="test query",
            )

            result = _build_ranker(mock_store, query, num_searches=2)

            mock_vectorsearch.VertexRanker.assert_called_once_with(
                query="test query",
                model="semantic-ranker-default@latest",
                title_template="{{ title }}",
                content_template="{{ content }}",
            )
            mock_vectorsearch.Ranker.assert_called_once()
            assert result == mock_ranker
