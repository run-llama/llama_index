"""Test Vertex AI Vector Store Vector Search functionality."""

import hashlib
import importlib.util
import logging
import os
import uuid
from collections.abc import Iterator
from typing import Any, Literal
from unittest.mock import MagicMock, call, patch

import pytest
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from pydantic import ValidationError

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import (
    Document,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore, utils
from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
    VectorSearchSDKManager,
)
from llama_index.vector_stores.vertexaivectorsearch._types import (
    VertexAIAddError,
    VertexAIDeleteError,
    VertexAIInputError,
)
from llama_index.vector_stores.vertexaivectorsearch.base import (
    FeatureFlags,
)
from llama_index.vector_stores.vertexaivectorsearch.utils import (
    calculate_rrf_weights,
    convert_filters_to_v2_format,
    retry,
)

V2_AVAILABLE = importlib.util.find_spec("google.cloud.vectorsearch_v1beta") is not None
xfail_if_missing_v2 = pytest.mark.xfail(
    condition=not V2_AVAILABLE, reason="requires v2 support"
)
xpass_if_missing_v2 = pytest.mark.xpass(
    condition=V2_AVAILABLE, reason="requires v2 support"
)

if V2_AVAILABLE:
    from google.cloud import vectorsearch_v1beta
    from google.cloud.vectorsearch_v1beta import (
        BatchCreateDataObjectsRequest,
        BatchDeleteDataObjectsRequest,
        BatchSearchDataObjectsRequest,
        BatchSearchDataObjectsResponse,
        BatchUpdateDataObjectsRequest,
        Collection,
        CreateDataObjectRequest,
        DataObject,
        DataObjectSearchServiceAsyncClient,
        DataObjectSearchServiceClient,
        DataObjectServiceAsyncClient,
        DataObjectServiceClient,
        DeleteDataObjectRequest,
        DenseVector,
        OutputFields,
        QueryDataObjectsRequest,
        QueryDataObjectsResponse,
        SearchDataObjectsRequest,
        SearchDataObjectsResponse,
        SearchResult,
        SparseVector,
        UpdateDataObjectRequest,
        Vector,
        VectorSearchServiceClient,
    )
    from google.cloud.vectorsearch_v1beta.services.data_object_search_service.pagers import (
        QueryDataObjectsAsyncPager,
        QueryDataObjectsPager,
        SearchDataObjectsAsyncPager,
        SearchDataObjectsPager,
    )

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

    def test_add_documents(self, node_embeddings: list[TextNode]) -> None:
        """Test adding documents to Vertex AI Vector Search vector store."""
        vector_store = self.vector_store()

        # Add nodes to the Vertex AI Vector Search index
        input_doc_ids = [node_embedding.id_ for node_embedding in node_embeddings]
        doc_ids = vector_store.add(node_embeddings)

        # Ensure that all nodes are returned & they are the same as input
        assert len(doc_ids) == len(node_embeddings)
        for doc_id in doc_ids:
            assert doc_id in input_doc_ids

    def test_search(self, node_embeddings: list[TextNode]) -> None:
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

    def test_search_with_filter(self, node_embeddings: list[TextNode]) -> None:
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

    def test_batch_update_index(self, node_embeddings: list[TextNode]) -> None:
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

    @pytest.fixture(autouse=True)
    def mock_sdk_manager(self) -> Iterator[None]:
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            yield

    def test_v1_requires_index_id(self):
        """Test that v1 raises error when index_id is missing."""
        with pytest.raises(
            ValueError, match=r".index_id. is required for api_version=.v1."
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                endpoint_id="projects/test/locations/us-central1/indexEndpoints/123",
            )

    def test_v1_requires_endpoint_id(self):
        """Test that v1 raises error when endpoint_id is missing."""
        with pytest.raises(
            ValueError, match=r".endpoint_id. is required for api_version=.v1."
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v1",
                index_id="projects/test/locations/us-central1/indexes/123",
            )

    def test_v1_rejects_collection_id(self):
        """Test that v1 raises error when collection_id is provided."""
        with pytest.raises(
            ValueError, match=r".collection_id. is only valid for api_version=.v2."
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
        with pytest.raises(
            ValueError, match=r".collection_id. is required for api_version=.v2."
        ):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
            )

    def test_v2_rejects_index_id(self):
        """Test that v2 raises error when index_id is provided."""
        with pytest.raises(
            ValueError, match=".index_id. is only valid for api_version=.v1."
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
            ValueError, match=".endpoint_id. is only valid for api_version=.v1."
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


class TestV2FeatureFlags:
    """Test feature flag behavior for v2."""

    def test_should_use_v2_with_v2_enabled(self):
        """Test that should_use_v2 returns True when api_version='v2' and flag enabled."""
        with patch.object(FeatureFlags, "ENABLE_V2", True):
            assert FeatureFlags.should_use_v2("v2") is True

    def test_should_use_v2_with_v2_disabled(self):
        """Test that should_use_v2 returns False when flag is disabled."""
        with patch.object(FeatureFlags, "ENABLE_V2", False):
            assert FeatureFlags.should_use_v2("v2") is False

    def test_should_use_v2_with_v1_version(self):
        """Test that should_use_v2 returns False when api_version='v1'."""
        with patch.object(FeatureFlags, "ENABLE_V2", True):
            assert FeatureFlags.should_use_v2("v1") is False


class TestV2SDKManager:
    """Test SDK manager v2 client functionality."""

    @xfail_if_missing_v2
    def test_ensure_v2_available_passes_when_sdk_installed(self) -> None:
        """Test that VectorSearchSDKManager.ensure_v2_available succeeds for tests."""
        manager = VectorSearchSDKManager(
            project_id="test-project", region="us-central1"
        )
        manager.ensure_v2_available()

    @xpass_if_missing_v2
    def test_ensure_v2_available_fails_when_sdk_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that VectorSearchSDKManager.ensure_v2_available raises an appropriate error."""
        manager = VectorSearchSDKManager(
            project_id="test-project", region="us-central1"
        )
        monkeypatch.setattr(manager, "_v2_available", False)

        with pytest.raises(
            ImportError, match=r"Vertex v2 operations require the .v2. extra"
        ):
            manager.ensure_v2_available()

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


class TestV2RetryDecorator:
    """Test the retry decorator for v2 operations."""

    def test_retry_succeeds_on_first_attempt(self):
        """Test that function returns immediately on success."""
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

    @pytest.fixture(autouse=True)
    def mock_sdk_manager(self) -> Iterator[None]:
        with patch(
            "llama_index.vector_stores.vertexaivectorsearch.base.VectorSearchSDKManager"
        ):
            yield

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
        with pytest.raises(ValidationError):
            VertexAIVectorStore(
                project_id="test-project",
                region="us-central1",
                api_version="v2",
                collection_id="my-collection",
                default_hybrid_alpha=-0.1,
            )

    def test_hybrid_ranker_must_be_rrf_or_vertex(self):
        """Test that hybrid_ranker must be 'rrf'."""
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
        )

        assert store.enable_hybrid is True
        assert store.text_search_fields == ["title", "content"]
        assert store.embedding_field == "embedding"
        assert store.hybrid_ranker == "rrf"
        assert store.default_hybrid_alpha == 0.7
        assert store.semantic_task_type == "RETRIEVAL_QUERY"


class TestV2FilterConversion:
    """Test LlamaIndex filter to V2 filter conversion."""

    def test_simple_eq_filter(self):
        """Test simple equality filter conversion."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            ]
        )

        result = convert_filters_to_v2_format(filters)

        assert result == {"category": {"$eq": "tech"}}

    def test_gt_filter(self):
        """Test greater than filter conversion."""
        filters = MetadataFilters(
            filters=[MetadataFilter(key="price", value=50, operator=FilterOperator.GT)]
        )

        result = convert_filters_to_v2_format(filters)

        assert result == {"price": {"$gt": 50}}

    def test_and_filter(self):
        """Test AND filter conversion."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="tech", operator=FilterOperator.EQ
                ),
                MetadataFilter(key="price", value=50, operator=FilterOperator.LT),
            ],
            condition=FilterCondition.AND,
        )

        result = convert_filters_to_v2_format(filters)

        assert result == {
            "$and": [
                {"category": {"$eq": "tech"}},
                {"price": {"$lt": 50}},
            ]
        }

    def test_or_filter(self):
        """Test OR filter conversion."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="color", value="red", operator=FilterOperator.EQ),
                MetadataFilter(key="color", value="blue", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )

        result = convert_filters_to_v2_format(filters)

        assert result == {
            "$or": [
                {"color": {"$eq": "red"}},
                {"color": {"$eq": "blue"}},
            ]
        }

    def test_in_filter(self):
        """Test IN filter conversion."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tags", value=["ml", "ai"], operator=FilterOperator.IN
                )
            ]
        )

        result = convert_filters_to_v2_format(filters)

        assert result == {"tags": {"$in": ["ml", "ai"]}}

    def test_none_filters(self):
        """Test that None filters returns None."""
        result = convert_filters_to_v2_format(None)
        assert result is None

    def test_empty_filters(self):
        """Test that empty filters returns None."""
        result = convert_filters_to_v2_format(MetadataFilters(filters=[]))
        assert result is None


# Common mocks for V2 service clients


@pytest.fixture
def mock_v2_collection() -> "Collection":
    return Collection(
        {
            "name": V2_COLLECTION_PARENT,
            "create_time": "2026-04-01T10:00:00.000Z",
            "update_time": "2026-04-02T11:00:00.000Z",
            "data_schema": {
                "type": "object",
                "required": ["text", "user_id"],
                "properties": {
                    "publish_time": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                    "title": {"type": "string"},
                    "user_id": {"type": "number"},
                },
            },
            "vector_schema": {
                "embedding": {"dense_vector": {"dimensions": 1024}},
                "sparse_embedding": {"sparse_vector": {}},
                "semantic_embedding": {
                    "dense_vector": {
                        "dimensions": 1024,
                        "vertex_embedding_config": {
                            "model_id": "model1",
                            "text_template": "{text}",
                            "task_type": "RETRIEVAL_QUERY",
                        },
                    },
                },
            },
        }
    )


@pytest.fixture
def mock_v2_vector_search_service_client(mock_v2_collection: "Collection") -> MagicMock:
    return MagicMock(
        spec=VectorSearchServiceClient,
        get_collection=MagicMock(return_value=mock_v2_collection),
    )


@pytest.fixture
def mock_v2_data_object_service_client() -> MagicMock:
    return MagicMock(
        spec=DataObjectServiceClient,
        batch_create_data_objects=MagicMock(return_value=MagicMock()),
    )


@pytest.fixture
def mock_v2_data_object_service_async_client() -> MagicMock:
    return MagicMock(spec=DataObjectServiceAsyncClient)


@pytest.fixture
def mock_v2_data_object_search_service_client() -> MagicMock:
    return MagicMock(spec=DataObjectSearchServiceClient)


@pytest.fixture
def mock_v2_data_object_search_service_async_client() -> MagicMock:
    return MagicMock(spec=DataObjectSearchServiceAsyncClient)


@pytest.fixture
def mock_v2_service_clients(
    mock_v2_vector_search_service_client: MagicMock,
    mock_v2_data_object_service_client: MagicMock,
    mock_v2_data_object_service_async_client: MagicMock,
    mock_v2_data_object_search_service_client: MagicMock,
    mock_v2_data_object_search_service_async_client: MagicMock,
) -> Iterator[None]:
    with (
        patch.object(vectorsearch_v1beta, "VectorSearchServiceClient") as vs_cls,
        patch.object(vectorsearch_v1beta, "DataObjectServiceClient") as do_cls,
        patch.object(vectorsearch_v1beta, "DataObjectServiceAsyncClient") as ado_cls,
        patch.object(vectorsearch_v1beta, "DataObjectSearchServiceClient") as ds_cls,
        patch.object(
            vectorsearch_v1beta, "DataObjectSearchServiceAsyncClient"
        ) as ads_cls,
    ):
        vs_cls.return_value = mock_v2_vector_search_service_client
        do_cls.return_value = mock_v2_data_object_service_client
        ado_cls.return_value = mock_v2_data_object_service_async_client
        ds_cls.return_value = mock_v2_data_object_search_service_client
        ads_cls.return_value = mock_v2_data_object_search_service_async_client
        yield


@pytest.fixture
def input_dense_nodes() -> list[TextNode]:
    # corresponds to `output_dense_data_objects`
    return [
        TextNode(
            id_=f"node_{i}",
            text=f"Text {i}",
            embedding=[i / 100 for _ in range(4)],
            # for ref_doc_id / source_node
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"doc_{i // 2}")
            },
            metadata={
                "title": f"Title {i}",
                "user_id": i * 100,
                "title_embedding": [i / 200 for _ in range(4)],
                "sparse_embedding": {
                    "indices": [2, 4, 6, 8],
                    "values": [0.2, 0.4, 0.6, 0.8],
                },
            },
        )
        for i in range(5)
    ]


@pytest.fixture
def output_dense_data_objects() -> list["DataObject"]:
    # corresponds to `input_dense_nodes`
    return [
        DataObject(
            data={
                "node_id": f"node_{i}",
                "text": f"Text {i}",
                "title": f"Title {i}",
                "user_id": i * 100,
                "node_type": "TextNode",
                "parent_id": f"doc_{i // 2}",
            },
            vectors={
                "embedding": Vector(
                    dense=DenseVector(values=[i / 100 for _ in range(4)])
                ),
                "title_embedding": Vector(
                    dense=DenseVector(values=[i / 200 for _ in range(4)])
                ),
                "sparse_embedding": Vector(
                    sparse=SparseVector(
                        indices=[2, 4, 6, 8], values=[0.2, 0.4, 0.6, 0.8]
                    )
                ),
            },
        )
        for i in range(5)
    ]


V2_COLLECTION_PARENT = (
    "projects/test-project/locations/us-central1/collections/my-collection"
)


@xfail_if_missing_v2
class TestUnitV2NodeDataObjectConversion:
    """Test bidirectional conversion between ``DataObject`` and ``TextNode``."""

    @pytest.fixture
    def mock_v2_store(
        self, mock_v2_service_clients: Iterator[None], **kwargs: Any
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            default_add_operation="create",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
        )

    def test_extract_v2_data_object_from_node(
        self,
        mock_v2_store: VertexAIVectorStore,
        input_dense_nodes: list[TextNode],
        output_dense_data_objects: list["DataObject"],
    ) -> None:
        """Test round-trip conversion of ``DataObject``s."""
        # WHEN
        data_objects_from_nodes = [
            mock_v2_store.extract_v2_data_object_from_node(node)
            for node in input_dense_nodes
        ]

        # THEN
        assert data_objects_from_nodes == output_dense_data_objects

    def test_extract_node_from_v2_data_object(
        self,
        mock_v2_store: VertexAIVectorStore,
        input_dense_nodes: list[TextNode],
        output_dense_data_objects: list["DataObject"],
    ) -> None:
        """Test round-trip conversion of ``TextNode``s."""
        # WHEN
        nodes_from_data_objects = [
            mock_v2_store.extract_node_from_v2_data_object(do)
            for do in output_dense_data_objects
        ]

        # THEN
        assert len(nodes_from_data_objects) == len(input_dense_nodes)
        for actual, expected in zip(
            nodes_from_data_objects, input_dense_nodes, strict=False
        ):
            assert actual.node_id == expected.node_id
            assert isinstance(actual, TextNode)
            assert actual.text == expected.text
            assert actual.relationships == expected.relationships
            assert pytest.approx(actual.embedding) == expected.embedding
            assert actual.metadata.keys() == expected.metadata.keys()
            actual_title_embed = actual.metadata.pop("title_embedding")
            assert (
                pytest.approx(actual_title_embed)
                == expected.metadata["title_embedding"]
            )
            actual_sparse = actual.metadata.pop("sparse_embedding")
            assert (
                actual_sparse["indices"]
                == expected.metadata["sparse_embedding"]["indices"]
            )
            assert (
                pytest.approx(actual_sparse["values"])
                == expected.metadata["sparse_embedding"]["values"]
            )
            assert actual.metadata == {
                k: v for k, v in expected.metadata.items() if k in actual.metadata
            }


@pytest.mark.parametrize(
    (
        "vector_store_fixture_name",
        "requests_fixture_name",
        "extra_args",
        "add_operation_type",
    ),
    [
        (
            "mock_v2_store_add_create",
            "expected_add_create_requests",
            {},
            "create",
        ),
        (
            "mock_v2_store_add_update",
            "expected_add_create_requests",
            {"add_operation": "create"},
            "create",
        ),
        (
            "mock_v2_store_add_update",
            "expected_add_update_requests",
            {},
            "update",
        ),
        (
            "mock_v2_store_add_create",
            "expected_add_update_requests",
            {"add_operation": "update"},
            "update",
        ),
    ],
    ids=[
        "add_operation=CREATE (via default)",
        "add_operation=CREATE (via override)",
        "add_operation=UPDATE (via default)",
        "add_operation=UPDATE (via override)",
    ],
)
@xfail_if_missing_v2
class TestUnitV2Add:
    """Test the behavior of ``add`` and ``async_add`` methods."""

    @pytest.fixture
    def mock_v2_store_add_create(
        self, mock_v2_service_clients: Iterator[None], **kwargs: Any
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            default_add_operation="create",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
        )

    @pytest.fixture
    def mock_v2_store_add_update(
        self, mock_v2_service_clients: Iterator[None], **kwargs: Any
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            default_add_operation="update",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
        )

    @pytest.fixture
    def output_dense_create_data_object_requests(
        self,
        output_dense_data_objects: list["DataObject"],
        **kwargs: Any,
    ) -> list["CreateDataObjectRequest"]:
        # corresponds to `input_dense_nodes`
        return [
            CreateDataObjectRequest(
                parent=V2_COLLECTION_PARENT,
                data_object_id=f"node_{i}",
                data_object=obj,
            )
            for i, obj in enumerate(output_dense_data_objects)
        ]

    @pytest.fixture
    def output_dense_update_data_object_requests(
        self,
        output_dense_data_objects: list["DataObject"],
        **kwargs: Any,
    ) -> list["UpdateDataObjectRequest"]:
        # corresponds to `input_dense_nodes` for UPDATE operations
        return [
            UpdateDataObjectRequest(
                data_object=DataObject(
                    name=f"{V2_COLLECTION_PARENT}/dataObjects/node_{i}",
                    data=obj.data,
                    vectors=obj.vectors,
                )
            )
            for i, obj in enumerate(output_dense_data_objects)
        ]

    @pytest.fixture
    def expected_add_create_requests(
        self,
        output_dense_create_data_object_requests: list["CreateDataObjectRequest"],
        **kwargs: Any,
    ) -> list["BatchCreateDataObjectsRequest"]:
        return [
            BatchCreateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_create_data_object_requests[0:2],
            ),
            BatchCreateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_create_data_object_requests[2:4],
            ),
            BatchCreateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_create_data_object_requests[4:],
            ),
        ]

    @pytest.fixture
    def expected_add_update_requests(
        self,
        output_dense_update_data_object_requests: list["UpdateDataObjectRequest"],
        **kwargs: Any,
    ) -> list["BatchUpdateDataObjectsRequest"]:
        return [
            BatchUpdateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_update_data_object_requests[0:2],
            ),
            BatchUpdateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_update_data_object_requests[2:4],
            ),
            BatchUpdateDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=output_dense_update_data_object_requests[4:],
            ),
        ]

    def test_v2_add_valid(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        mock_v2_data_object_service_client: MagicMock,
        input_dense_nodes: list[TextNode],
    ) -> None:
        """Test that appropriate calls are made with prepared data objects when ``add`` is called."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)
        expected_requests = request.getfixturevalue(requests_fixture_name)
        expected_calls = [call(request) for request in expected_requests]
        expected_output_ids = [f"node_{i}" for i in range(5)]

        # WHEN
        actual_output_ids = vector_store.add(input_dense_nodes, **extra_args)

        # THEN
        assert actual_output_ids == expected_output_ids
        match add_operation_type:
            case "create":
                mock_v2_data_object_service_client.batch_create_data_objects.assert_has_calls(
                    expected_calls,
                )
                mock_v2_data_object_service_client.batch_update_data_objects.assert_not_called()
            case "update":
                mock_v2_data_object_service_client.batch_create_data_objects.assert_not_called()
                mock_v2_data_object_service_client.batch_update_data_objects.assert_has_calls(
                    expected_calls
                )
            case _:
                raise ValueError("Update the test!")

    async def test_v2_async_add_valid(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        mock_v2_data_object_service_async_client: MagicMock,
        input_dense_nodes: list[TextNode],
    ) -> None:
        """Test that appropriate calls are made with prepared data objects when ``async_add`` is called."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)
        expected_requests = request.getfixturevalue(requests_fixture_name)
        expected_calls = [call(request) for request in expected_requests]
        expected_output_ids = [f"node_{i}" for i in range(5)]

        # WHEN
        actual_output_ids = await vector_store.async_add(
            input_dense_nodes, **extra_args
        )

        # THEN
        assert sorted(actual_output_ids) == expected_output_ids
        match add_operation_type:
            case "create":
                mock_v2_data_object_service_async_client.batch_create_data_objects.assert_has_awaits(
                    expected_calls, any_order=True
                )
                mock_v2_data_object_service_async_client.batch_update_data_objects.assert_not_awaited()
            case "update":
                mock_v2_data_object_service_async_client.batch_create_data_objects.assert_not_awaited()
                mock_v2_data_object_service_async_client.batch_update_data_objects.assert_has_awaits(
                    expected_calls, any_order=True
                )
            case _:
                raise ValueError("Update the test!")

    def test_v2_add_empty(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        mock_v2_data_object_service_client: MagicMock,
    ) -> None:
        """Test that no calls are made when an empty list of nodes is passed to ``add``."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)

        # WHEN
        actual_output_ids = vector_store.add([], **extra_args)

        # THEN
        assert actual_output_ids == []
        mock_v2_data_object_service_client.batch_create_data_objects.assert_not_called()
        mock_v2_data_object_service_client.batch_update_data_objects.assert_not_called()

    async def test_v2_async_add_empty(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        mock_v2_data_object_service_async_client: MagicMock,
    ) -> None:
        """Test that no calls are made when an empty list of nodes is passed to ``async_add``."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)

        # WHEN
        actual_output_ids = await vector_store.async_add([], **extra_args)

        # THEN
        assert actual_output_ids == []
        mock_v2_data_object_service_async_client.batch_create_data_objects.assert_not_awaited()
        mock_v2_data_object_service_async_client.batch_update_data_objects.assert_not_awaited()

    def test_v2_add_invalid_parameters(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        input_dense_nodes: list[TextNode],
    ) -> None:
        """Test that an error is raised for invalid parameters in v2 ``add``."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)

        # WHEN / THEN
        with pytest.raises(
            ValueError,
            match=r".is_complete_overwrite. is only valid for api_version=.v1.",
        ):
            _ = vector_store.add(input_dense_nodes, is_complete_overwrite=True)

    async def test_v2_async_add_invalid_parameters(
        self,
        request: pytest.FixtureRequest,
        vector_store_fixture_name: str,
        requests_fixture_name: str,
        extra_args: dict[str, Any],
        add_operation_type: Literal["create", "update"],
        input_dense_nodes: list[TextNode],
    ) -> None:
        """Test that an error is raised for invalid parameters in v2 ``async_add``."""
        # GIVEN
        vector_store = request.getfixturevalue(vector_store_fixture_name)

        # WHEN / THEN
        with pytest.raises(
            ValueError,
            match=r".is_complete_overwrite. is only valid for api_version=.v1.",
        ):
            _ = await vector_store.async_add(
                input_dense_nodes, is_complete_overwrite=True
            )

    @pytest.mark.parametrize(
        ("input_metadata", "error_msg"),
        [
            (
                {"title_embedding": 0.5},
                "Invalid dense embedding field 'title_embedding'",
            ),
            (
                {"sparse_embedding": [0.1, 0.2, 0.3]},
                "Invalid sparse embedding field 'sparse_embedding'",
            ),
        ],
        ids=["invalid dense vector format", "invalid sparse vector format"],
    )
    class TestLogsInvalidDataObjects:
        def test_v2_add_logs_invalid_data_objects(
            self,
            caplog: pytest.LogCaptureFixture,
            request: pytest.FixtureRequest,
            vector_store_fixture_name: str,
            requests_fixture_name: str,
            extra_args: dict[str, Any],
            add_operation_type: Literal["create", "update"],
            input_dense_nodes: list[TextNode],
            input_metadata: dict[str, Any],
            error_msg: str,
        ) -> None:
            """Test that appropriate errors logs are made for badly structured data."""
            # GIVEN
            vector_store = request.getfixturevalue(vector_store_fixture_name)
            input_nodes = [
                TextNode(
                    id_="node_1",
                    text="Text 1",
                    embedding=[0.1, 0.2, 0.3],
                    metadata=input_metadata,
                )
            ]

            # WHEN
            with caplog.at_level(logging.ERROR):
                _ = vector_store.add(input_nodes, **extra_args)

            # THEN
            assert error_msg in caplog.text

        async def test_v2_async_add_logs_invalid_data_objects(
            self,
            caplog: pytest.LogCaptureFixture,
            request: pytest.FixtureRequest,
            vector_store_fixture_name: str,
            requests_fixture_name: str,
            extra_args: dict[str, Any],
            add_operation_type: Literal["create", "update"],
            input_dense_nodes: list[TextNode],
            input_metadata: dict[str, Any],
            error_msg: str,
        ) -> None:
            """Test that appropriate errors logs are made for badly structured data."""
            # GIVEN
            vector_store = request.getfixturevalue(vector_store_fixture_name)
            input_nodes = [
                TextNode(
                    id_="node_1",
                    text="Text 1",
                    embedding=[0.1, 0.2, 0.3],
                    metadata=input_metadata,
                )
            ]

            # WHEN
            with caplog.at_level(logging.ERROR):
                _ = await vector_store.async_add(input_nodes, **extra_args)

            # THEN
            assert error_msg in caplog.text

    @pytest.mark.parametrize(
        ("batch_add_side_effect", "expected_changed_ids", "expected_failed_ids"),
        [
            ([ValueError, None, None], [2, 3, 4], [0, 1]),
            ([None, ValueError, None], [0, 1, 4], [2, 3]),
            ([None, None, ValueError], [0, 1, 2, 3], [4]),
            ([ValueError, ValueError, None], [4], [0, 1, 2, 3]),
            ([None, ValueError, ValueError], [0, 1], [2, 3, 4]),
            ([ValueError, None, ValueError], [2, 3], [0, 1, 4]),
            ([ValueError, ValueError, ValueError], [], [0, 1, 2, 3, 4]),
        ],
        ids=[
            "1 exc, pos=1",
            "1 exc, pos=2",
            "1 exc, pos=3",
            "2 excs, pos=1,2",
            "2 excs, pos=2,3",
            "2 excs, pos=1,3",
            "3 excs, pos=1,2,3",
        ],
    )
    class TestSubBatchFailures:
        """Tests for when a subset of add batches fail, with common test parameterization."""

        def test_v2_add_failed_sub_requests(
            self,
            request: pytest.FixtureRequest,
            vector_store_fixture_name: str,
            requests_fixture_name: str,
            extra_args: dict[str, Any],
            add_operation_type: Literal["create", "update"],
            mock_v2_data_object_service_client: MagicMock,
            input_dense_nodes: list[TextNode],
            batch_add_side_effect: list[Exception | None],
            expected_changed_ids: list[int],
            expected_failed_ids: list[int],
        ) -> None:
            """Test that the appropriate exception is raised when a subset of ``add`` batches fail."""
            # GIVEN
            vector_store = request.getfixturevalue(vector_store_fixture_name)
            expected_requests = request.getfixturevalue(requests_fixture_name)
            expected_calls = [call(request) for request in expected_requests]
            expected_changed = [f"node_{i}" for i in expected_changed_ids]
            expected_failed = [f"node_{i}" for i in expected_failed_ids]
            match add_operation_type:
                case "create":
                    mock_v2_data_object_service_client.batch_create_data_objects.side_effect = batch_add_side_effect
                    mock_v2_data_object_service_client.batch_update_data_objects.side_effect = ValueError(
                        "wrong request!"
                    )
                case "update":
                    mock_v2_data_object_service_client.batch_create_data_objects.side_effect = ValueError(
                        "wrong request!"
                    )
                    mock_v2_data_object_service_client.batch_update_data_objects.side_effect = batch_add_side_effect

            # WHEN
            with pytest.raises(VertexAIAddError) as exc_info:
                _ = vector_store.add(input_dense_nodes, **extra_args)

            # THEN
            exception = exc_info.value
            assert isinstance(exception, VertexAIAddError)
            match add_operation_type:
                case "create":
                    assert exception.result.added_ids == expected_changed
                    assert exception.result.failed_ids == expected_failed
                    mock_v2_data_object_service_client.batch_create_data_objects.assert_has_calls(
                        expected_calls,
                    )
                    mock_v2_data_object_service_client.batch_update_data_objects.assert_not_called()
                case "update":
                    assert exception.result.updated_ids == expected_changed
                    assert exception.result.failed_ids == expected_failed
                    mock_v2_data_object_service_client.batch_create_data_objects.assert_not_called()
                    mock_v2_data_object_service_client.batch_update_data_objects.assert_has_calls(
                        expected_calls
                    )
                case _:
                    raise ValueError("Update the test!")

        async def test_v2_async_add_failed_sub_requests(
            self,
            request: pytest.FixtureRequest,
            vector_store_fixture_name: str,
            requests_fixture_name: str,
            extra_args: dict[str, Any],
            add_operation_type: Literal["create", "update"],
            mock_v2_data_object_service_async_client: MagicMock,
            input_dense_nodes: list[TextNode],
            batch_add_side_effect: list[Exception | None],
            expected_changed_ids: list[int],
            expected_failed_ids: list[int],
        ) -> None:
            """Test that the appropriate exception is raised when a subset of ``async_add`` batches fail."""
            # GIVEN
            vector_store = request.getfixturevalue(vector_store_fixture_name)
            expected_requests = request.getfixturevalue(requests_fixture_name)
            expected_calls = [call(request) for request in expected_requests]
            expected_changed = [f"node_{i}" for i in expected_changed_ids]
            expected_failed = [f"node_{i}" for i in expected_failed_ids]
            match add_operation_type:
                case "create":
                    mock_v2_data_object_service_async_client.batch_create_data_objects.side_effect = batch_add_side_effect
                    mock_v2_data_object_service_async_client.batch_update_data_objects.side_effect = ValueError(
                        "wrong request!"
                    )
                case "update":
                    mock_v2_data_object_service_async_client.batch_create_data_objects.side_effect = ValueError(
                        "wrong request!"
                    )
                    mock_v2_data_object_service_async_client.batch_update_data_objects.side_effect = batch_add_side_effect

            # WHEN
            with pytest.raises(VertexAIAddError) as exc_info:
                _ = await vector_store.async_add(input_dense_nodes, **extra_args)

            # THEN
            exception = exc_info.value
            assert isinstance(exception, VertexAIAddError)
            match add_operation_type:
                case "create":
                    assert exception.result.added_ids == expected_changed
                    assert exception.result.failed_ids == expected_failed
                    mock_v2_data_object_service_async_client.batch_create_data_objects.assert_has_awaits(
                        expected_calls, any_order=True
                    )
                    mock_v2_data_object_service_async_client.batch_update_data_objects.assert_not_awaited()
                case "update":
                    assert exception.result.updated_ids == expected_changed
                    assert exception.result.failed_ids == expected_failed
                    mock_v2_data_object_service_async_client.batch_create_data_objects.assert_not_awaited()
                    mock_v2_data_object_service_async_client.batch_update_data_objects.assert_has_awaits(
                        expected_calls, any_order=True
                    )
                case _:
                    raise ValueError("Update the test!")


@xfail_if_missing_v2
class TestUnitV2Delete:
    """Unit test the behavior of ``(a)delete``, ``(a)delete_nodes``, and ``(a)clear``."""

    @pytest.fixture
    def mock_v2_store(
        self, mock_v2_service_clients: Iterator[None]
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
        )

    @pytest.fixture
    def expected_delete_requests_ref_id_1(
        self,
    ) -> list["BatchDeleteDataObjectsRequest"]:
        return [
            BatchDeleteDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=[
                    DeleteDataObjectRequest(
                        name=f"{V2_COLLECTION_PARENT}/dataObjects/node_2"
                    ),
                ],
            ),
            BatchDeleteDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=[
                    DeleteDataObjectRequest(
                        name=f"{V2_COLLECTION_PARENT}/dataObjects/node_3"
                    ),
                ],
            ),
        ]

    @pytest.fixture
    def delete_ref_id_query_pager_pages_ref_id_1(self) -> list[MagicMock]:
        return [
            MagicMock(
                spec=QueryDataObjectsResponse,
                data_objects=[
                    DataObject(
                        name=f"{V2_COLLECTION_PARENT}/dataObjects/node_2",
                        data={"node_id": "node_2"},
                    )
                ],
            ),
            MagicMock(
                spec=QueryDataObjectsResponse,
                data_objects=[
                    DataObject(
                        name=f"{V2_COLLECTION_PARENT}/dataObjects/node_3",
                        data={"node_id": "node_3"},
                    )
                ],
            ),
        ]

    @pytest.fixture
    def expected_delete_nodes_by_id_requests(
        self,
    ) -> list["BatchDeleteDataObjectsRequest"]:
        # batch_size=2 → two batches: ["node_0","node_1"] and ["node_2"]
        def _req(nids: list[str]) -> BatchDeleteDataObjectsRequest:
            return BatchDeleteDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                requests=[
                    DeleteDataObjectRequest(
                        name=f"{V2_COLLECTION_PARENT}/dataObjects/{nid}",
                    )
                    for nid in nids
                ],
            )

        return [_req(["node_0", "node_1"]), _req(["node_2"])]

    @pytest.fixture
    def input_delete_nodes_filters(self) -> MetadataFilters:
        # corresponds to `expected_filter_query_request`
        return MetadataFilters(filters=[MetadataFilter(key="user_id", value=200)])

    @pytest.fixture
    def expected_filter_query_request(self) -> "QueryDataObjectsRequest":
        # corresponds to `input_delete_nodes_filters`
        return QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"user_id": {"$eq": 200}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )

    def test_v2_delete_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
        )
        input_ref_doc_id = "doc_1"
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"parent_id": {"$eq": input_ref_doc_id}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        mock_v2_store.delete(ref_doc_id=input_ref_doc_id)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    async def test_v2_adelete_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = (
            delete_ref_id_query_pager_pages_ref_id_1
        )
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
        input_ref_doc_id = "doc_1"
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"parent_id": {"$eq": input_ref_doc_id}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        await mock_v2_store.adelete(ref_doc_id=input_ref_doc_id)

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    def test_v2_delete_missing_docid_field(
        self, mock_v2_store: VertexAIVectorStore
    ) -> None:
        # GIVEN
        mock_v2_store.docid_field = None
        input_ref_doc_id = "doc_1"

        # WHEN / THEN
        with pytest.raises(ValueError):
            mock_v2_store.delete(ref_doc_id=input_ref_doc_id)

    async def test_v2_adelete_missing_docid_field(
        self, mock_v2_store: VertexAIVectorStore
    ) -> None:
        # GIVEN
        mock_v2_store.docid_field = None
        input_ref_doc_id = "doc_1"

        # WHEN / THEN
        with pytest.raises(ValueError):
            await mock_v2_store.adelete(ref_doc_id=input_ref_doc_id)

    def test_v2_delete_empty_query_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(spec=QueryDataObjectsPager, pages=[])
        )
        input_ref_doc_id = "doc_1"
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"parent_id": {"$eq": input_ref_doc_id}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )

        # WHEN
        mock_v2_store.delete(ref_doc_id=input_ref_doc_id)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_not_called()

    async def test_v2_adelete_empty_query_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = []
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
        input_ref_doc_id = "doc_1"
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"parent_id": {"$eq": input_ref_doc_id}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )

        # WHEN
        await mock_v2_store.adelete(ref_doc_id=input_ref_doc_id)

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_not_called()

    @pytest.mark.parametrize(
        "input_node_ids", [[], None], ids=["node_ids=[]", "node_ids=None"]
    )
    def test_v2_delete_nodes_invalid_input(
        self,
        mock_v2_store: VertexAIVectorStore,
        input_node_ids: list[str] | None,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            mock_v2_store.delete_nodes(node_ids=input_node_ids, filters=None)

    @pytest.mark.parametrize(
        "input_node_ids", [[], None], ids=["node_ids=[]", "node_ids=None"]
    )
    async def test_v2_adelete_nodes_invalid_input(
        self,
        mock_v2_store: VertexAIVectorStore,
        input_node_ids: list[str] | None,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            await mock_v2_store.adelete_nodes(node_ids=input_node_ids, filters=None)

    def test_v2_delete_nodes_by_id_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        expected_delete_nodes_by_id_requests: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        expected_calls = [call(req) for req in expected_delete_nodes_by_id_requests]

        # WHEN
        mock_v2_store.delete_nodes(node_ids=["node_0", "node_1", "node_2"])

        # THEN
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    async def test_v2_adelete_nodes_by_id_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        expected_delete_nodes_by_id_requests: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        expected_calls = [call(req) for req in expected_delete_nodes_by_id_requests]

        # WHEN
        await mock_v2_store.adelete_nodes(node_ids=["node_0", "node_1", "node_2"])

        # THEN
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
            expected_calls, any_order=True
        )

    def test_v2_delete_nodes_by_filters_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        input_delete_nodes_filters: MetadataFilters,
        expected_filter_query_request: "QueryDataObjectsRequest",
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
        )
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        mock_v2_store.delete_nodes(filters=input_delete_nodes_filters)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_filter_query_request,
        )
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    async def test_v2_adelete_nodes_by_filters_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
        input_delete_nodes_filters: MetadataFilters,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_filter_query_request: "QueryDataObjectsRequest",
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = (
            delete_ref_id_query_pager_pages_ref_id_1
        )
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        await mock_v2_store.adelete_nodes(filters=input_delete_nodes_filters)

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_filter_query_request
        )
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    def test_v2_delete_nodes_by_filters_empty_query_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
        input_delete_nodes_filters: MetadataFilters,
        expected_filter_query_request: "QueryDataObjectsRequest",
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(spec=QueryDataObjectsPager, pages=[])
        )

        # WHEN
        mock_v2_store.delete_nodes(filters=input_delete_nodes_filters)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_filter_query_request
        )
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_not_called()

    async def test_v2_adelete_nodes_by_filters_empty_query_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
        input_delete_nodes_filters: MetadataFilters,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_filter_query_request: "QueryDataObjectsRequest",
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = []
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager

        # WHEN
        await mock_v2_store.adelete_nodes(filters=input_delete_nodes_filters)

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_filter_query_request
        )
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_not_called()

    def test_v2_delete_nodes_by_filters_valid_empty_filters(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # GIVEN
        caplog.set_level(logging.WARNING)
        input_filters = MetadataFilters(filters=[])

        # WHEN
        mock_v2_store.delete_nodes(filters=input_filters)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_not_called()
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_not_called()
        assert "Input filter set is empty after conversion" in caplog.text

    async def test_v2_adelete_nodes_by_filters_valid_empty_filters(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # GIVEN
        caplog.set_level(logging.WARNING)
        input_filters = MetadataFilters(filters=[])

        # WHEN
        await mock_v2_store.adelete_nodes(filters=input_filters)

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_not_called()
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_not_called()
        assert "Input filter set is empty after conversion" in caplog.text

    def test_v2_clear_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_client: MagicMock,
        mock_v2_data_object_search_service_client: MagicMock,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
        )
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        mock_v2_store.clear()

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    async def test_v2_aclear_valid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_service_async_client: MagicMock,
        mock_v2_data_object_search_service_async_client: MagicMock,
        delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
        expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = (
            delete_ref_id_query_pager_pages_ref_id_1
        )
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]

        # WHEN
        await mock_v2_store.aclear()

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
            expected_calls,
        )

    @pytest.mark.parametrize(
        (
            "batch_delete_side_effect",
            "expected_results",
            "delete_by_id_expected_results",
        ),
        [
            ([NotFound("not found"), None], (1, 1, 0), (1, 2, 0)),
            ([None, NotFound("not found")], (1, 1, 0), (2, 1, 0)),
            ([NotFound("not found"), NotFound("not found")], (0, 2, 0), (0, 3, 0)),
            ([ValueError(), None], (1, 0, 1), (1, 0, 2)),
            ([None, ValueError()], (1, 0, 1), (2, 0, 1)),
            ([ValueError(), ValueError()], (0, 0, 2), (0, 0, 3)),
            ([NotFound("not found"), ValueError()], (0, 1, 1), (0, 2, 1)),
            ([ValueError(), NotFound("not found")], (0, 1, 1), (0, 1, 2)),
        ],
        ids=[
            "1 NotFound (1st batch)",
            "1 NotFound (2nd batch)",
            "2 NotFound",
            "1 general error (1st batch)",
            "1 general error (2nd batch)",
            "2 general errors",
            "NotFound + general error",
            "general error + NotFound",
        ],
    )
    class TestSubBatchFailures:
        """Tests for when a subset of delete batches fail, with common test parameterization."""

        def test_v2_delete_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_client: MagicMock,
            mock_v2_data_object_search_service_client: MagicMock,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_v2_data_object_search_service_client.query_data_objects.return_value = MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
            input_ref_doc_id = "doc_1"
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"parent_id": {"$eq": input_ref_doc_id}},
                page_size=2,
                output_fields=OutputFields(metadata_fields=["*"]),
            )
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_client.batch_delete_data_objects.side_effect = (
                batch_delete_side_effect
            )
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                mock_v2_store.delete(ref_doc_id=input_ref_doc_id)

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        async def test_v2_adelete_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_async_client: MagicMock,
            mock_v2_data_object_search_service_async_client: MagicMock,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
            mock_pager.pages.__aiter__.return_value = (
                delete_ref_id_query_pager_pages_ref_id_1
            )
            mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
            input_ref_doc_id = "doc_1"
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"parent_id": {"$eq": input_ref_doc_id}},
                page_size=2,
                output_fields=OutputFields(metadata_fields=["*"]),
            )
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_async_client.batch_delete_data_objects.side_effect = batch_delete_side_effect
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                await mock_v2_store.adelete(ref_doc_id=input_ref_doc_id)

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        def test_v2_delete_nodes_by_id_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_client: MagicMock,
            expected_delete_nodes_by_id_requests: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            expected_calls = [call(req) for req in expected_delete_nodes_by_id_requests]
            mock_v2_data_object_service_client.batch_delete_data_objects.side_effect = (
                batch_delete_side_effect
            )
            expected_deleted, expected_not_found, expected_failed = (
                delete_by_id_expected_results
            )
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                mock_v2_store.delete_nodes(node_ids=["node_0", "node_1", "node_2"])

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        async def test_v2_adelete_nodes_by_id_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_async_client: MagicMock,
            expected_delete_nodes_by_id_requests: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            expected_calls = [call(req) for req in expected_delete_nodes_by_id_requests]
            mock_v2_data_object_service_async_client.batch_delete_data_objects.side_effect = batch_delete_side_effect
            expected_deleted, expected_not_found, expected_failed = (
                delete_by_id_expected_results
            )
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                await mock_v2_store.adelete_nodes(
                    node_ids=["node_0", "node_1", "node_2"]
                )

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        def test_v2_delete_nodes_by_filters_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_client: MagicMock,
            mock_v2_data_object_search_service_client: MagicMock,
            input_delete_nodes_filters: MetadataFilters,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_filter_query_request: "QueryDataObjectsRequest",
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_v2_data_object_search_service_client.query_data_objects.return_value = MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_client.batch_delete_data_objects.side_effect = (
                batch_delete_side_effect
            )
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                mock_v2_store.delete_nodes(filters=input_delete_nodes_filters)

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
                expected_filter_query_request,
            )
            mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        async def test_v2_adelete_nodes_by_filters_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_async_client: MagicMock,
            mock_v2_data_object_search_service_async_client: MagicMock,
            input_delete_nodes_filters: MetadataFilters,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_filter_query_request: "QueryDataObjectsRequest",
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
            mock_pager.pages.__aiter__.return_value = (
                delete_ref_id_query_pager_pages_ref_id_1
            )
            mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_async_client.batch_delete_data_objects.side_effect = batch_delete_side_effect
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                await mock_v2_store.adelete_nodes(filters=input_delete_nodes_filters)

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
                expected_filter_query_request
            )
            mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        def test_v2_clear_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_client: MagicMock,
            mock_v2_data_object_search_service_client: MagicMock,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_v2_data_object_search_service_client.query_data_objects.return_value = MagicMock(
                spec=QueryDataObjectsPager,
                pages=delete_ref_id_query_pager_pages_ref_id_1,
            )
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                page_size=2,
                output_fields=OutputFields(metadata_fields=["*"]),
            )
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_client.batch_delete_data_objects.side_effect = (
                batch_delete_side_effect
            )
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                mock_v2_store.clear()

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            mock_v2_data_object_service_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )

        async def test_v2_aclear_failed_sub_request(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_service_async_client: MagicMock,
            mock_v2_data_object_search_service_async_client: MagicMock,
            delete_ref_id_query_pager_pages_ref_id_1: list[MagicMock],
            expected_delete_requests_ref_id_1: list["BatchDeleteDataObjectsRequest"],
            batch_delete_side_effect: list[Exception | None],
            expected_results: tuple[int, int, int],
            delete_by_id_expected_results: tuple[int, int, int],
        ) -> None:
            # GIVEN
            mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
            mock_pager.pages.__aiter__.return_value = (
                delete_ref_id_query_pager_pages_ref_id_1
            )
            mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                page_size=2,
                output_fields=OutputFields(metadata_fields=["*"]),
            )
            expected_calls = [call(req) for req in expected_delete_requests_ref_id_1]
            mock_v2_data_object_service_async_client.batch_delete_data_objects.side_effect = batch_delete_side_effect
            expected_deleted, expected_not_found, expected_failed = expected_results
            expected_exceptions = [e for e in batch_delete_side_effect if e is not None]

            # WHEN
            with pytest.raises(VertexAIDeleteError) as exc_info:
                await mock_v2_store.aclear()

            # THEN
            raised_exc = exc_info.value
            assert raised_exc.result.deleted == expected_deleted
            assert raised_exc.result.not_found == expected_not_found
            assert raised_exc.result.failed == expected_failed
            assert raised_exc.result.exceptions == expected_exceptions
            mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            mock_v2_data_object_service_async_client.batch_delete_data_objects.assert_has_calls(
                expected_calls,
            )


@xfail_if_missing_v2
class TestUnitV2GetNodes:
    """Test the behavior of ``get_nodes`` and ``aget_nodes`` methods."""

    @pytest.fixture
    def mock_v2_store(
        self, mock_v2_service_clients: Iterator[None]
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
        )

    @pytest.fixture
    def get_nodes_query_result_data_objects(self) -> list["DataObject"]:
        """DataObjects with ``name`` set, as returned by ``query_data_objects``."""
        return [
            DataObject(
                name=f"{V2_COLLECTION_PARENT}/node_{i}",
                data={"_node_content": f"Content {i}", "title": f"Title {i}"},
            )
            for i in range(2)
        ]

    @pytest.fixture
    def get_nodes_result_pages(
        self, get_nodes_query_result_data_objects: list["DataObject"]
    ) -> list[MagicMock]:
        return [
            MagicMock(
                spec=QueryDataObjectsResponse,
                data_objects=[obj],
            )
            for obj in get_nodes_query_result_data_objects
        ]

    @pytest.mark.parametrize(
        ("get_nodes_output_fields", "expected_output_fields"),
        [
            (None, {"metadata_fields": ["*"]}),
            (
                {"metadata_fields": ["*"], "data_fields": ["title"]},
                {"metadata_fields": ["*"], "data_fields": ["title"]},
            ),
        ],
        ids=["vector store default", "modified fields"],
    )
    class TestValidInput:
        def test_v2_get_nodes_by_id_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_client: MagicMock,
            get_nodes_query_result_data_objects: list["DataObject"],
            get_nodes_result_pages: list[MagicMock],
            get_nodes_output_fields: dict[str, list[str]] | None,
            expected_output_fields: dict[str, list[str]],
        ) -> None:
            # GIVEN
            if get_nodes_output_fields is not None:
                mock_v2_store.get_nodes_output_fields = get_nodes_output_fields
            mock_v2_data_object_search_service_client.query_data_objects.return_value = MagicMock(
                spec=QueryDataObjectsPager, pages=get_nodes_result_pages
            )
            input_node_ids = ["node_0", "node_1"]
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"object_id": {"$in": input_node_ids}},
                page_size=2,
                output_fields=expected_output_fields,
            )

            # WHEN
            result = mock_v2_store.get_nodes(node_ids=input_node_ids)

            # THEN
            mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            assert len(result) == len(get_nodes_query_result_data_objects)
            assert [node.node_id for node in result] == ["node_0", "node_1"]

        async def test_v2_aget_nodes_by_id_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_async_client: MagicMock,
            get_nodes_query_result_data_objects: list["DataObject"],
            get_nodes_result_pages: list[MagicMock],
            get_nodes_output_fields: dict[str, list[str]] | None,
            expected_output_fields: dict[str, list[str]],
        ) -> None:
            # GIVEN
            if get_nodes_output_fields is not None:
                mock_v2_store.get_nodes_output_fields = get_nodes_output_fields
            mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
            mock_pager.pages.__aiter__.return_value = get_nodes_result_pages
            mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
            input_node_ids = ["node_0", "node_1"]
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"object_id": {"$in": input_node_ids}},
                page_size=2,
                output_fields=expected_output_fields,
            )

            # WHEN
            result = await mock_v2_store.aget_nodes(
                node_ids=input_node_ids,
            )

            # THEN
            mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            assert len(result) == len(get_nodes_query_result_data_objects)
            assert [node.node_id for node in result] == ["node_0", "node_1"]

        def test_v2_get_nodes_by_filters_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_client: MagicMock,
            get_nodes_query_result_data_objects: list["DataObject"],
            get_nodes_result_pages: list[MagicMock],
            get_nodes_output_fields: dict[str, list[str]] | None,
            expected_output_fields: dict[str, list[str]],
        ) -> None:
            # GIVEN
            if get_nodes_output_fields is not None:
                mock_v2_store.get_nodes_output_fields = get_nodes_output_fields
            mock_v2_data_object_search_service_client.query_data_objects.return_value = MagicMock(
                spec=QueryDataObjectsPager, pages=get_nodes_result_pages
            )
            input_filters = MetadataFilters(
                filters=[MetadataFilter(key="user_id", value=200)]
            )
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"user_id": {"$eq": 200}},
                page_size=2,
                output_fields=expected_output_fields,
            )

            # WHEN
            result = mock_v2_store.get_nodes(filters=input_filters)

            # THEN
            mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            assert len(result) == len(get_nodes_query_result_data_objects)
            assert [node.node_id for node in result] == ["node_0", "node_1"]

        async def test_v2_aget_nodes_by_filters_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_async_client: MagicMock,
            get_nodes_query_result_data_objects: list["DataObject"],
            get_nodes_result_pages: list[MagicMock],
            get_nodes_output_fields: dict[str, list[str]] | None,
            expected_output_fields: dict[str, list[str]],
        ) -> None:
            # GIVEN
            if get_nodes_output_fields is not None:
                mock_v2_store.get_nodes_output_fields = get_nodes_output_fields
            mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
            mock_pager.pages.__aiter__.return_value = get_nodes_result_pages
            mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
            input_filters = MetadataFilters(
                filters=[MetadataFilter(key="user_id", value=200)]
            )
            expected_query_request = QueryDataObjectsRequest(
                parent=V2_COLLECTION_PARENT,
                filter={"user_id": {"$eq": 200}},
                page_size=2,
                output_fields=expected_output_fields,
            )

            # WHEN
            result = await mock_v2_store.aget_nodes(
                filters=input_filters,
            )

            # THEN
            mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
                expected_query_request,
            )
            assert len(result) == len(get_nodes_query_result_data_objects)
            assert [node.node_id for node in result] == ["node_0", "node_1"]

    def test_v2_get_nodes_both_inputs_invalid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_client: MagicMock,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            mock_v2_store.get_nodes(
                node_ids=["node_0"],
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="user_id", value=200)]
                ),
            )
        mock_v2_data_object_search_service_client.query_data_objects.assert_not_called()

    async def test_v2_aget_nodes_both_inputs_invalid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_async_client: MagicMock,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            await mock_v2_store.aget_nodes(
                node_ids=["node_0"],
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="user_id", value=200)]
                ),
            )
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_not_called()

    @pytest.mark.parametrize(
        "input_node_ids", [[], None], ids=["node_ids=[]", "node_ids=None"]
    )
    def test_v2_get_nodes_neither_input_invalid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_client: MagicMock,
        input_node_ids: list[str] | None,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            mock_v2_store.get_nodes(node_ids=input_node_ids, filters=None)
        mock_v2_data_object_search_service_client.query_data_objects.assert_not_called()

    @pytest.mark.parametrize(
        "input_node_ids", [[], None], ids=["node_ids=[]", "node_ids=None"]
    )
    async def test_v2_aget_nodes_neither_input_invalid(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_async_client: MagicMock,
        input_node_ids: list[str] | None,
    ) -> None:
        # WHEN / THEN
        with pytest.raises(ValueError):
            await mock_v2_store.aget_nodes(node_ids=input_node_ids, filters=None)
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_not_called()

    def test_v2_get_nodes_empty_filters(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # GIVEN
        caplog.set_level(logging.WARNING)

        # WHEN
        result = mock_v2_store.get_nodes(
            filters=MetadataFilters(filters=[]),
        )

        # THEN
        assert result == []
        mock_v2_data_object_search_service_client.query_data_objects.assert_not_called()
        assert "Input filter set is empty after conversion" in caplog.text

    async def test_v2_aget_nodes_empty_filters(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_async_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # GIVEN
        caplog.set_level(logging.WARNING)

        # WHEN
        result = await mock_v2_store.aget_nodes(
            filters=MetadataFilters(filters=[]),
        )

        # THEN
        assert result == []
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_not_called()
        assert "Input filter set is empty after conversion" in caplog.text

    def test_v2_get_nodes_empty_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_client: MagicMock,
    ) -> None:
        # GIVEN
        mock_v2_data_object_search_service_client.query_data_objects.return_value = (
            MagicMock(spec=QueryDataObjectsPager, pages=[])
        )
        input_node_ids = ["node_0"]
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"object_id": {"$in": input_node_ids}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )

        # WHEN
        result = mock_v2_store.get_nodes(node_ids=input_node_ids)

        # THEN
        mock_v2_data_object_search_service_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        assert result == []

    async def test_v2_aget_nodes_empty_result(
        self,
        mock_v2_store: VertexAIVectorStore,
        mock_v2_data_object_search_service_async_client: MagicMock,
    ) -> None:
        # GIVEN
        mock_pager = MagicMock(spec=QueryDataObjectsAsyncPager)
        mock_pager.pages.__aiter__.return_value = []
        mock_v2_data_object_search_service_async_client.query_data_objects.return_value = mock_pager
        input_node_ids = ["node_0"]
        expected_query_request = QueryDataObjectsRequest(
            parent=V2_COLLECTION_PARENT,
            filter={"object_id": {"$in": input_node_ids}},
            page_size=2,
            output_fields=OutputFields(metadata_fields=["*"]),
        )

        # WHEN
        result = await mock_v2_store.aget_nodes(
            node_ids=input_node_ids,
        )

        # THEN
        mock_v2_data_object_search_service_async_client.query_data_objects.assert_called_with(
            expected_query_request,
        )
        assert result == []


DEFAULT_OUTPUT_FIELDS = {"data_fields": ["*"], "metadata_fields": ["*"]}
CUSTOM_OUTPUT_FIELDS = {
    "data_fields": ["text", "title"],
    "vector_fields": ["embedding"],
    "metadata_fields": ["*"],
}
INPUT_FILTERS = MetadataFilters(
    filters=[MetadataFilter(key="some_key", value=3, operator=FilterOperator.GT)]
)
DENSE_SEARCH_BASIC = {
    "search_field": "embedding",
    "vector": {"values": [0.1, 0.2, 0.3, 0.4]},
    "top_k": 3,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
DENSE_SEARCH_FILTERS = {
    **DENSE_SEARCH_BASIC,
    "search_field": "other_embedding",
    "filter": {"some_key": {"$gt": 3}},
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
SPARSE_SEARCH_BASIC_K3 = {
    "search_field": "sparse_embedding",
    "sparse_vector": {"indices": [1, 2, 3, 4], "values": [0.1, 0.2, 0.3, 0.4]},
    "top_k": 3,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
SPARSE_SEARCH_BASIC_K5 = {
    **SPARSE_SEARCH_BASIC_K3,
    "top_k": 5,
}
SPARSE_SEARCH_FILTERS = {
    **SPARSE_SEARCH_BASIC_K3,
    "search_field": "other_sparse_embedding",
    "filter": {"some_key": {"$gt": 3}},
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
TEXT_SEARCH_BASIC = {
    "search_text": "my search query",
    "data_field_names": ["text", "title"],
    "top_k": 5,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
TEXT_SEARCH_FILTERS_K3 = {
    **TEXT_SEARCH_BASIC,
    "top_k": 3,
    "filter": {"some_key": {"$gt": 3}},
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
TEXT_SEARCH_FILTERS_K5 = {
    **TEXT_SEARCH_BASIC,
    "filter": {"some_key": {"$gt": 3}},
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
SEMANTIC_SEARCH_BASIC = {
    "search_text": "my search query",
    "search_field": "semantic_embedding",
    "task_type": "RETRIEVAL_QUERY",
    "top_k": 3,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
SEMANTIC_SEARCH_FILTERS = {
    **SEMANTIC_SEARCH_BASIC,
    "search_field": "semantic_embedding",
    "filter": {"some_key": {"$gt": 3}},
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
RRF_RANKER_1 = {
    "ranker": {"rrf": {"weights": [1.0]}},
    "top_k": 6,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
RRF_RANKER_2 = {
    "ranker": {"rrf": {"weights": [0.5, 0.5]}},
    "top_k": 6,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}
RRF_RANKER_2_CUSTOM_FIELDS = {
    "ranker": {"rrf": {"weights": [0.5, 0.5]}},
    "top_k": 6,
    "output_fields": CUSTOM_OUTPUT_FIELDS,
}
RRF_RANKER_3 = {
    "ranker": {"rrf": {"weights": [1 / 3, 1 / 3, 1 / 3]}},
    "top_k": 6,
    "output_fields": DEFAULT_OUTPUT_FIELDS,
}


@xfail_if_missing_v2
class TestUnitV2Query:
    """Test behavior of ``query`` and ``aquery`` methods."""

    @pytest.fixture
    def mock_v2_store(
        self, mock_v2_service_clients: Iterator[None]
    ) -> VertexAIVectorStore:
        return VertexAIVectorStore(
            project_id="test-project",
            region="us-central1",
            api_version="v2",
            collection_id="my-collection",
            nodeid_field="node_id",
            node_type_field="node_type",
            docid_field="parent_id",
            content_field="text",
            batch_size=2,
            max_concurrent_requests=5,
            text_search_fields=["text", "title"],
            embedding_field="embedding",
            sparse_embedding_field="sparse_embedding",
            semantic_search_embedding_field="semantic_embedding",
            dense_embedding_fields={"title_embedding"},
            sparse_embedding_fields={"sparse_embedding"},
            enable_hybrid=True,
        )

    @pytest.fixture
    def output_dense_search_results(self) -> list["SearchResult"]:
        return [
            SearchResult(
                data_object=DataObject(
                    data={
                        "node_id": f"node_{i}",
                        "text": f"Text {i}",
                        "title": f"Title {i}",
                        "user_id": i * 100,
                        "node_type": "TextNode",
                        "parent_id": f"doc_{i // 2}",
                    },
                ),
                distance=score,
            )
            for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])
        ]

    @pytest.fixture
    def expected_query_result(self) -> VectorStoreQueryResult:
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        nodes = [
            TextNode(
                id_=f"node_{i}",
                text=f"Text {i}",
                metadata={
                    "title": f"Title {i}",
                    "user_id": i * 100,
                },
                embedding=None,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"doc_{i // 2}")
                },
            )
            for i in range(5)
        ]
        ids = [node.id_ for node in nodes]
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    @pytest.mark.parametrize(
        ("input_query", "expected_search", "query_kwargs", "uses_batch_request"),
        [
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    similarity_top_k=3,
                ),
                {"parent": V2_COLLECTION_PARENT, "vector_search": DENSE_SEARCH_BASIC},
                {},
                False,
                id="mode=default (dense), embedding_field=store, filter=no",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    embedding_field="other_embedding",
                    similarity_top_k=3,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "vector_search": DENSE_SEARCH_FILTERS,
                },
                {},
                False,
                id="mode=default (dense), embedding_field=query, filter=yes",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_str="my search query",
                    similarity_top_k=3,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "semantic_search": SEMANTIC_SEARCH_BASIC,
                },
                {},
                False,
                id="mode=default (semantic), embedding_field=store-semantic, filter=no",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_str="my search query",
                    embedding_field="embedding",
                    similarity_top_k=3,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "semantic_search": SEMANTIC_SEARCH_FILTERS,
                },
                {},
                False,
                id="mode=default (semantic), embedding_field=query, filter=yes",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SPARSE,
                    query_embedding=None,
                    sparse_top_k=3,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "vector_search": SPARSE_SEARCH_BASIC_K3,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                False,
                id="mode=sparse, embedding_field=store, top_k=sparse, filter=no",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SPARSE,
                    embedding_field="other_sparse_embedding",
                    similarity_top_k=3,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "vector_search": SPARSE_SEARCH_FILTERS,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                False,
                id="mode=sparse, embedding_field=query, top_k=similarity, filter=yes",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                    query_str="my search query",
                    sparse_top_k=5,
                    similarity_top_k=3,
                ),
                {"parent": V2_COLLECTION_PARENT, "text_search": TEXT_SEARCH_BASIC},
                {},
                False,
                id="mode=text_search, filter=no, top_k=sparse",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                    query_str="my search query",
                    similarity_top_k=3,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {"parent": V2_COLLECTION_PARENT, "text_search": TEXT_SEARCH_FILTERS_K3},
                {},
                False,
                id="mode=text_search, filter=yes, top_k=similarity",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_BASIC},
                        {"vector_search": DENSE_SEARCH_BASIC},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {},
                True,
                id="mode=hybrid, vector_search=dense, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    embedding_field="other_embedding",
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_FILTERS_K5},
                        {"vector_search": DENSE_SEARCH_FILTERS},
                    ],
                    "combine": RRF_RANKER_2_CUSTOM_FIELDS,
                },
                {},
                True,
                id="mode=hybrid, vector_search=dense, embedding_field=query, filter=yes, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_BASIC},
                        {"semantic_search": SEMANTIC_SEARCH_BASIC},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {},
                True,
                id="mode=hybrid, vector_search=semantic, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    embedding_field="other_embedding",
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_FILTERS_K5},
                        {"semantic_search": SEMANTIC_SEARCH_FILTERS},
                    ],
                    "combine": RRF_RANKER_2_CUSTOM_FIELDS,
                },
                {},
                True,
                id="mode=hybrid, vector_search=semantic, embedding_field=query, filter=yes, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_BASIC},
                        {"vector_search": DENSE_SEARCH_BASIC},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {},
                True,
                id="mode=hybrid, vector_search=dense, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_BASIC},
                        {"vector_search": SPARSE_SEARCH_BASIC_K5},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                True,
                id="mode=hybrid, vector_search=sparse, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"text_search": TEXT_SEARCH_BASIC},
                        {"vector_search": DENSE_SEARCH_BASIC},
                        {"vector_search": SPARSE_SEARCH_BASIC_K5},
                    ],
                    "combine": RRF_RANKER_3,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                True,
                id="mode=hybrid, vector_search=dense+sparse, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"semantic_search": SEMANTIC_SEARCH_BASIC},
                        {"vector_search": DENSE_SEARCH_BASIC},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {},
                True,
                id="mode=semantic_hybrid, vector_search=dense, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my search query",
                    sparse_top_k=3,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"semantic_search": SEMANTIC_SEARCH_BASIC},
                        {"vector_search": SPARSE_SEARCH_BASIC_K3},
                    ],
                    "combine": RRF_RANKER_2,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                True,
                id="mode=semantic_hybrid, vector_search=sparse, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my search query",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"semantic_search": SEMANTIC_SEARCH_BASIC},
                        {"vector_search": DENSE_SEARCH_BASIC},
                        {"vector_search": SPARSE_SEARCH_BASIC_K5},
                    ],
                    "combine": RRF_RANKER_3,
                },
                {"sparse_embedding": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}},
                True,
                id="mode=semantic_hybrid, vector_search=dense+sparse, embedding_field=store, filter=no, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my search query",
                    embedding_field="other_embedding",
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                    filters=INPUT_FILTERS,
                    output_fields=["text", "title", "embedding"],
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [
                        {"semantic_search": SEMANTIC_SEARCH_FILTERS},
                        {"vector_search": DENSE_SEARCH_FILTERS},
                    ],
                    "combine": RRF_RANKER_2_CUSTOM_FIELDS,
                },
                {},
                True,
                id="mode=semantic_hybrid, vector_search=dense, embedding_field=query, filter=yes, top_k=hybrid",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my search query",
                    sparse_top_k=5,
                    similarity_top_k=3,
                    hybrid_top_k=6,
                ),
                {
                    "parent": V2_COLLECTION_PARENT,
                    "searches": [{"semantic_search": SEMANTIC_SEARCH_BASIC}],
                    "combine": RRF_RANKER_1,
                },
                {},
                True,
                id="mode=semantic_hybrid, vector_search=none, embedding_field=store, filter=no, top_k=hybrid",
            ),
        ],
    )
    class TestValidInput:
        def test_v2_query_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_client: MagicMock,
            input_query: VectorStoreQuery,
            expected_search: dict[str, Any],
            query_kwargs: dict[str, Any],
            uses_batch_request: bool,
            output_dense_search_results: list["SearchResult"],
            expected_query_result: VectorStoreQueryResult,
        ) -> None:
            # GIVEN
            if uses_batch_request:
                raw_results = MagicMock(
                    spec=BatchSearchDataObjectsResponse,
                    results=[
                        MagicMock(
                            spec=SearchDataObjectsResponse,
                            results=output_dense_search_results,
                        )
                    ],
                )
                mock_v2_data_object_search_service_client.batch_search_data_objects.return_value = raw_results
            else:
                raw_results = MagicMock(spec=SearchDataObjectsPager)
                raw_results.__iter__.return_value = output_dense_search_results
                mock_v2_data_object_search_service_client.search_data_objects.return_value = raw_results

            # WHEN
            actual_query_result = mock_v2_store.query(input_query, **query_kwargs)

            # THEN
            assert actual_query_result == expected_query_result
            if uses_batch_request:
                mock_v2_data_object_search_service_client.batch_search_data_objects.assert_called_with(
                    BatchSearchDataObjectsRequest(expected_search)
                )
                mock_v2_data_object_search_service_client.search_data_objects.assert_not_called()
            else:
                mock_v2_data_object_search_service_client.search_data_objects.assert_called_with(
                    SearchDataObjectsRequest(expected_search)
                )
                mock_v2_data_object_search_service_client.batch_search_data_objects.assert_not_called()

        async def test_v2_aquery_valid(
            self,
            mock_v2_store: VertexAIVectorStore,
            mock_v2_data_object_search_service_async_client: MagicMock,
            input_query: VectorStoreQuery,
            expected_search: dict[str, Any],
            query_kwargs: dict[str, Any],
            uses_batch_request: bool,
            output_dense_search_results: list["SearchResult"],
            expected_query_result: VectorStoreQueryResult,
        ) -> None:
            # GIVEN
            if uses_batch_request:
                raw_results = MagicMock(
                    spec=BatchSearchDataObjectsResponse,
                    results=[
                        MagicMock(
                            spec=SearchDataObjectsResponse,
                            results=output_dense_search_results,
                        )
                    ],
                )
                mock_v2_data_object_search_service_async_client.batch_search_data_objects.return_value = raw_results
            else:
                raw_results = MagicMock(spec=SearchDataObjectsAsyncPager)
                raw_results.__aiter__.return_value = output_dense_search_results
                mock_v2_data_object_search_service_async_client.search_data_objects.return_value = raw_results

            # WHEN
            actual_query_result = await mock_v2_store.aquery(
                input_query, **query_kwargs
            )

            # THEN
            assert actual_query_result == expected_query_result
            if uses_batch_request:
                mock_v2_data_object_search_service_async_client.batch_search_data_objects.assert_called_with(
                    BatchSearchDataObjectsRequest(expected_search)
                )
                mock_v2_data_object_search_service_async_client.search_data_objects.assert_not_called()
            else:
                mock_v2_data_object_search_service_async_client.search_data_objects.assert_called_with(
                    SearchDataObjectsRequest(expected_search)
                )
                mock_v2_data_object_search_service_async_client.batch_search_data_objects.assert_not_called()

    @pytest.mark.parametrize(
        ("input_query", "vector_store_update_fields", "error_match"),
        [
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    similarity_top_k=3,
                ),
                {},
                r".*query_str. field must be set",
                id="mode=default, query_embedding=null, query_str=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_str="my search query",
                    similarity_top_k=3,
                ),
                {"semantic_search_embedding_field": "bad_field"},
                r"No valid auto-embedding field passed for semantic search",
                id="mode=default, store field doesn't support auto-embedding",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.DEFAULT,
                    query_str="my search query",
                    embedding_field="bad_field",
                    similarity_top_k=3,
                ),
                {"semantic_search_embedding_field": None},
                r"No valid auto-embedding field passed for semantic search",
                id="mode=default, no store auto-embedding, bad query embedding field",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SPARSE,
                    similarity_top_k=3,
                ),
                {},
                r"a .sparse_embedding. must be passed",
                id="mode=sparse, sparse_embedding=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    sparse_top_k=5,
                ),
                {},
                r".*query_str. field must be set",
                id="mode=text_search, query_str=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                    query_str="my query",
                    sparse_top_k=5,
                ),
                {"text_search_fields": None},
                r".*vector store field .text_search_fields. must be set",
                id="mode=text_search, store.text_search_fields=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my query",
                    hybrid_top_k=6,
                ),
                {"enable_hybrid": False},
                r".*vector store field .enable_hybrid. must be True",
                id="mode=hybrid, store.enable_hybrid=false",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    hybrid_top_k=6,
                ),
                {},
                r".*query_str. field must be set",
                id="mode=hybrid, query_str=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.HYBRID,
                    query_str="my query",
                    hybrid_top_k=6,
                ),
                {"text_search_fields": None},
                r".*vector store field .text_search_fields. must be set",
                id="mode=hybrid, store.text_search_fields=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_str="my query",
                    hybrid_top_k=6,
                ),
                {"enable_hybrid": False},
                r".*vector store field .enable_hybrid. must be True",
                id="mode=semantic_hybrid, store.enable_hybrid=false",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    hybrid_top_k=6,
                ),
                {},
                r".*query_str. field must be set",
                id="mode=semantic_hybrid, query_str=null",
            ),
            pytest.param(
                VectorStoreQuery(
                    mode=VectorStoreQueryMode.MMR,  # unsupported, falls back to DEFAULT
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                    similarity_top_k=3,
                ),
                {},
                "Unsupported query mode",
                id="mode=mmr, not supported",
            ),
        ],
    )
    class TestInvalidInput:
        def test_v2_query_invalid_input(
            self,
            mock_v2_store: VertexAIVectorStore,
            input_query: VectorStoreQuery,
            vector_store_update_fields: dict[str, Any],
            error_match: str,
        ) -> None:
            # GIVEN
            if vector_store_update_fields:
                for key, val in vector_store_update_fields.items():
                    setattr(mock_v2_store, key, val)

            # WHEN / THEN
            with pytest.raises(VertexAIInputError, match=error_match):
                _ = mock_v2_store.query(input_query)

        async def test_v2_aquery_invalid_input(
            self,
            mock_v2_store: VertexAIVectorStore,
            input_query: VectorStoreQuery,
            vector_store_update_fields: dict[str, Any],
            error_match: str,
        ) -> None:
            # GIVEN
            if vector_store_update_fields:
                for key, val in vector_store_update_fields.items():
                    setattr(mock_v2_store, key, val)

            # WHEN / THEN
            with pytest.raises(VertexAIInputError, match=error_match):
                _ = await mock_v2_store.aquery(input_query)

    @pytest.mark.parametrize(
        ("alpha", "num_searches", "expected"),
        [
            (0.0, 2, [0.0, 1.0]),
            (1.0, 2, [1.0, 0.0]),
            (0.5, 2, [0.5, 0.5]),
            (0.7, 2, [pytest.approx(0.7), pytest.approx(0.3)]),
            (0.5, 3, [pytest.approx(1.0 / 3) for _ in range(3)]),
        ],
        ids=[
            "alpha=0.0 gives pure text weight",
            "alpha=1.0 gives pure vector weight",
            "alpha=0.5 gives balanced weights",
            "alpha=0.7 favors vector search",
            "3 searches get equal weights when not two",
        ],
    )
    def test_rrf_weight_calculation(
        self, alpha: float, num_searches: int, expected: list[float]
    ) -> None:
        weights = calculate_rrf_weights(alpha=alpha, num_searches=num_searches)
        assert weights == expected
