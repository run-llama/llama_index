"""Test Vertex AI Vector Store Vector Search functionality."""

import os
import uuid
import hashlib

from typing import List

import pytest

from llama_index.core.schema import MetadataMode, TextNode, Document
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
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


def test_class():
    names_of_base_classes = [b.__name__ for b in VertexAIVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
