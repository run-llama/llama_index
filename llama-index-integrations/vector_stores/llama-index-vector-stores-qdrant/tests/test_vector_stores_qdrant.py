from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.core.schema import TextNode
import pytest

from qdrant_client.http.models import PointsList, PointStruct


def test_class():
    names_of_base_classes = [b.__name__ for b in QdrantVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_delete__and_get_nodes(vector_store: QdrantVectorStore) -> None:
    vector_store.delete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = vector_store.get_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


def test_clear(vector_store: QdrantVectorStore) -> None:
    vector_store.clear()
    with pytest.raises(ValueError, match="Collection test not found"):
        vector_store.get_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )


@pytest.mark.asyncio()
async def test_adelete_and_aget(vector_store: QdrantVectorStore) -> None:
    await vector_store.adelete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = await vector_store.aget_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


@pytest.mark.asyncio()
async def test_aclear(vector_store: QdrantVectorStore) -> None:
    await vector_store.aclear()
    with pytest.raises(ValueError, match="Collection test not found"):
        await vector_store.aget_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )


def test_parse_query_result(vector_store: QdrantVectorStore) -> None:
    payload = {
        "text": "Hello, world!",
    }

    vector_dict = {
        "": [1, 2, 3],
    }

    # test vector name is empty (default)
    points = PointsList(points=[PointStruct(id=1, vector=vector_dict, payload=payload)])

    results = vector_store.parse_to_query_result(list(points.points))

    assert len(results.nodes) == 1
    assert results.nodes[0].embedding == [1, 2, 3]

    # test vector name is not empty
    vector_dict = {
        "text-dense": [1, 2, 3],
    }

    points = PointsList(points=[PointStruct(id=1, vector=vector_dict, payload=payload)])

    results = vector_store.parse_to_query_result(list(points.points))

    assert len(results.nodes) == 1
    assert results.nodes[0].embedding == [1, 2, 3]


@pytest.mark.asyncio()
async def test_get_with_embedding(vector_store: QdrantVectorStore) -> None:
    existing_nodes = await vector_store.aget_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )

    assert all(node.embedding is not None for node in existing_nodes)


class TestHybridQdrantVectorStore:
    hybrid_vector_store: QdrantVectorStore

    @pytest.fixture(autouse=True)
    def setup(self, hybrid_vector_store: QdrantVectorStore):
        # This will automatically assign the fixture to the class attribute
        self.hybrid_vector_store = hybrid_vector_store

    def test_get_sparse_embedding_query(self):
        """Check if the `_get_sparse_embedding` method returns the instance that is provided or not."""
        sparse_embedding = ([8, 12], [0.8, 0.12])
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="Test",
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
            sparse_query_embedding=sparse_embedding,
        )
        # Check for identity, we don't care about the exact numbers but need to know it is the same actual object
        assert sparse_embedding is self.hybrid_vector_store._get_sparse_embedding_query(
            query
        ), "The provided `sparse_query_embedding` is expected to be returned unmodified."

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="Test",
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
        )
        # Again, check or identity, otherwise the test could be fuzzy (even though highly unlikely) if the
        # `sparse_embedding_function` would ever return the same embedding that is predefined here.
        assert (
            sparse_embedding
            is not self.hybrid_vector_store._get_sparse_embedding_query(query)
        ), "No `sparse_query_embedding` is provided, we'd expect the `sparse_query_fn` to compute a fresh one."

    def test_get_sparse_embedding_nodes(self):
        # Case 1: All sparse embeddings are provided
        nodes = [
            TextNode(
                text="test1",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
                sparse_embedding=([3, 41], [0.3, 0.41]),
            ),
            TextNode(
                text="test2",
                id_="22222222-2222-2222-2222-222222222222",
                embedding=[0.0, 1.0],
                sparse_embedding=([2, 16], [0.2, 0.16]),
            ),
            TextNode(
                text="test3",
                id_="33333333-3333-3333-3333-333333333333",
                embedding=[1.0, 1.0],
                sparse_embedding=([7], [0.7]),
            ),
        ]
        sparse_embeddings = self.hybrid_vector_store._get_sparse_embeddings_nodes(nodes)
        assert sparse_embeddings == (
            [[3, 41], [2, 16], [7]],
            [[0.3, 0.41], [0.2, 0.16], [0.7]],
        )

        # Case 2: Some sparse embeddings are provided
        nodes = [
            TextNode(
                text="test1",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
                sparse_embedding=([3, 41], [0.3, 0.41]),
            ),
            TextNode(
                text="test2",
                id_="22222222-2222-2222-2222-222222222222",
            ),
            TextNode(
                text="test3",
                id_="33333333-3333-3333-3333-333333333333",
            ),
        ]
        sparse_embeddings = self.hybrid_vector_store._get_sparse_embeddings_nodes(nodes)
        assert sparse_embeddings[0][0] == [3, 41]
        assert sparse_embeddings[1][0] == [0.3, 0.41]
        # VERY unlikely, but these could be flaky if the sparse embedding model produces
        # the exact same embeddings.
        assert sparse_embeddings[0][1:] != [[2, 16], [7]]
        assert sparse_embeddings[1][1:] != [[0.2, 0.16], [0.7]]

        # Case 3: No sparse embeddings are provided
        nodes = [
            TextNode(
                text="test1",
                id_="11111111-1111-1111-1111-111111111111",
            ),
            TextNode(
                text="test2",
                id_="22222222-2222-2222-2222-222222222222",
            ),
            TextNode(
                text="test3",
                id_="33333333-3333-3333-3333-333333333333",
            ),
        ]
        sparse_embeddings = self.hybrid_vector_store._get_sparse_embeddings_nodes(nodes)
        assert sparse_embeddings != (
            [[3, 41], [2, 16], [7]],
            [[0.3, 0.41], [0.2, 0.16], [0.7]],
        )

    def test_query_dense(self):
        # This should trigger the fallback path (dense retrieval for hybrid collection)
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="test1",
            mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=1,
        )
        result = self.hybrid_vector_store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"

    def test_query_sparse(self):
        query = VectorStoreQuery(
            sparse_query_embedding=([3, 41], [0.3, 0.41]),
            query_str="test1",
            mode=VectorStoreQueryMode.SPARSE,
            similarity_top_k=1,
            sparse_top_k=1,
        )
        result = self.hybrid_vector_store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"

    def test_query_hybrid(self):
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            sparse_query_embedding=([3, 41], [0.3, 0.41]),
            query_str="test1",
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=1,
            hybrid_top_k=1,
        )
        result = self.hybrid_vector_store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"

    # Test Async Methods
    @pytest.mark.asyncio()
    async def test_aquery_dense(self):
        # This should trigger the fallback path (dense retrieval for hybrid collection)
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="test1",
            mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=1,
        )
        result = await self.hybrid_vector_store.aquery(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio()
    async def test_aquery_sparse(self):
        query = VectorStoreQuery(
            sparse_query_embedding=([3, 41], [0.3, 0.41]),
            query_str="test1",
            mode=VectorStoreQueryMode.SPARSE,
            similarity_top_k=1,
            sparse_top_k=1,
        )
        result = await self.hybrid_vector_store.aquery(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio()
    async def test_aquery_hybrid(self):
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            sparse_query_embedding=([3, 41], [0.3, 0.41]),
            query_str="test1",
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=1,
            hybrid_top_k=1,
        )
        result = await self.hybrid_vector_store.aquery(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "11111111-1111-1111-1111-111111111111"
