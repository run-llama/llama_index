import os
import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch, index

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
COLLECTION_NAME = "llama_index_test_hybrid"
VECTOR_INDEX_NAME = "vector_index_hybrid"
TEXT_INDEX_NAME = "text_index_hybrid"
DIM = 8
TIMEOUT = 120

@pytest.mark.skipif(
    MONGODB_URI is None,
    reason="Requires MONGODB_URI env variable for integration test",
)
class TestHybridIntegration:
    @pytest.fixture(scope="class")
    def vector_store(self) -> MongoDBAtlasVectorSearch:
        import pymongo
        client = pymongo.MongoClient(MONGODB_URI)
        return MongoDBAtlasVectorSearch(
            mongodb_client=client,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            vector_index_name=VECTOR_INDEX_NAME,
            fulltext_index_name=TEXT_INDEX_NAME,
        )

    @pytest.fixture(scope="class")
    def seeded(self, vector_store: MongoDBAtlasVectorSearch) -> bool:
        # Clear and seed predictable nodes
        clxn = vector_store.collection
        clxn.delete_many({})
        # Two groups: one more text-relevant, one more vector-similar
        nodes = [
            TextNode(text="alpha beta gamma", embedding=[0.9]*DIM, metadata={"group": "A"}),
            TextNode(text="alpha beta", embedding=[0.85]*DIM, metadata={"group": "A"}),
            TextNode(text="delta epsilon", embedding=[0.1]*DIM, metadata={"group": "B"}),
            TextNode(text="zeta eta theta", embedding=[0.2]*DIM, metadata={"group": "B"}),
        ]
        vector_store.add(nodes)
        return True

    @pytest.fixture(scope="class")
    def ensure_indexes(self, vector_store: MongoDBAtlasVectorSearch) -> bool:
        clxn = vector_store.collection
        # Create vector index if missing
        if not any(idx["name"] == VECTOR_INDEX_NAME for idx in clxn.list_search_indexes()):
            index.create_vector_search_index(
                collection=clxn,
                index_name=VECTOR_INDEX_NAME,
                dimensions=DIM,
                path="embedding",
                similarity="cosine",
                wait_until_complete=TIMEOUT,
            )
        # Create text index if missing
        if not any(idx["name"] == TEXT_INDEX_NAME for idx in clxn.list_search_indexes()):
            index.create_fulltext_search_index(
                collection=clxn,
                index_name=TEXT_INDEX_NAME,
                field="text",
                field_type="string",
                wait_until_complete=TIMEOUT,
            )
        return True

    def test_hybrid_alpha_bias_vector(self, vector_store: MongoDBAtlasVectorSearch, seeded: bool, ensure_indexes: bool) -> None:
        query_embedding = [0.9]*DIM  # Close to group A embeddings
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str="alpha",  # Matches first two more strongly by text
            similarity_top_k=3,
            hybrid_top_k=3,
            sparse_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.8,  # Bias towards vector similarity
        )
        res = vector_store.query(q)
        assert len(res.nodes) >= 2
        # With high alpha, vector similarity should dominate; group A expected early.
        groups = [n.metadata.get("group") for n in res.nodes]
        assert groups[0] == "A"

    def test_hybrid_alpha_bias_text(self, vector_store: MongoDBAtlasVectorSearch, seeded: bool, ensure_indexes: bool) -> None:
        query_embedding = [0.2]*DIM  # Closer to group B vectors
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str="alpha",  # Text strongly matches group A
            similarity_top_k=3,
            hybrid_top_k=3,
            sparse_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.2,  # Bias towards text relevance
        )
        res = vector_store.query(q)
        assert len(res.nodes) >= 2
        groups = [n.metadata.get("group") for n in res.nodes]
        # With low alpha, text match should dominate; group A still first.
        assert groups[0] == "A"
        # Ensure at least one B appears somewhere when embedding influences rank.
        assert any(g == "B" for g in groups)
