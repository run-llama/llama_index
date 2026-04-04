"""
Tests for Qdrant native BM25 support.

Native BM25 uses Qdrant's built-in BM25 tokenizer (server-side) instead of
requiring FastEmbed or other client-side sparse encoders. This sends
rest.Document objects to the server, which handles tokenization and scoring.

Since the in-memory QdrantClient does not support server-side Document
inference (it tries to fall back to fastembed), these tests mock the client
interactions to verify the integration logic is correct.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.base import (
    DEFAULT_NATIVE_BM25_MODEL,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME,
)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestNativeBm25Construction:
    """Verify constructor validation and field defaults for native BM25."""

    def _mock_client(self) -> MagicMock:
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        return client

    def test_enable_native_bm25_sets_hybrid_true(self):
        store = QdrantVectorStore(
            collection_name="test",
            client=self._mock_client(),
            enable_native_bm25=True,
        )
        assert store.enable_hybrid is True
        assert store.enable_native_bm25 is True
        assert store.native_bm25_model == DEFAULT_NATIVE_BM25_MODEL

    def test_custom_bm25_model_name(self):
        store = QdrantVectorStore(
            collection_name="test",
            client=self._mock_client(),
            enable_native_bm25=True,
            native_bm25_model="custom/bm25-multilingual",
        )
        assert store.native_bm25_model == "custom/bm25-multilingual"

    def test_bm25_config_is_stored(self):
        cfg = rest.Bm25Config(k=1.5, b=0.8, avg_len=200, language="german")
        store = QdrantVectorStore(
            collection_name="test",
            client=self._mock_client(),
            enable_native_bm25=True,
            bm25_config=cfg,
        )
        assert store._bm25_config is cfg

    def test_rejects_native_bm25_with_sparse_doc_fn(self):
        with pytest.raises(ValueError, match="Cannot use enable_native_bm25"):
            QdrantVectorStore(
                collection_name="test",
                client=self._mock_client(),
                enable_native_bm25=True,
                sparse_doc_fn=lambda texts: ([], []),
            )

    def test_rejects_native_bm25_with_sparse_query_fn(self):
        with pytest.raises(ValueError, match="Cannot use enable_native_bm25"):
            QdrantVectorStore(
                collection_name="test",
                client=self._mock_client(),
                enable_native_bm25=True,
                sparse_query_fn=lambda texts: ([], []),
            )

    def test_rejects_native_bm25_with_fastembed_model(self):
        with pytest.raises(ValueError, match="Cannot use enable_native_bm25"):
            QdrantVectorStore(
                collection_name="test",
                client=self._mock_client(),
                enable_native_bm25=True,
                fastembed_sparse_model="Qdrant/bm25",
            )

    def test_sparse_encoder_fns_are_none_when_native_bm25(self):
        store = QdrantVectorStore(
            collection_name="test",
            client=self._mock_client(),
            enable_native_bm25=True,
        )
        assert store._sparse_doc_fn is None
        assert store._sparse_query_fn is None
        assert store._hybrid_fusion_fn is not None

    def test_default_fields_when_native_bm25_disabled(self):
        """Sanity check: existing behaviour is preserved when native BM25 is off."""
        store = QdrantVectorStore(
            collection_name="test",
            client=self._mock_client(),
        )
        assert store.enable_native_bm25 is False
        assert store.enable_hybrid is False


# ---------------------------------------------------------------------------
# Document object building
# ---------------------------------------------------------------------------


class TestMakeBm25Document:
    """Verify _make_bm25_document produces correct rest.Document objects."""

    def _store(self, **kwargs) -> QdrantVectorStore:
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        return QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_native_bm25=True,
            **kwargs,
        )

    def test_default_model(self):
        store = self._store()
        doc = store._make_bm25_document("hello world")
        assert isinstance(doc, rest.Document)
        assert doc.text == "hello world"
        assert doc.model == DEFAULT_NATIVE_BM25_MODEL
        assert doc.options is None

    def test_custom_model(self):
        store = self._store(native_bm25_model="custom/model")
        doc = store._make_bm25_document("test")
        assert doc.model == "custom/model"

    def test_with_bm25_config(self):
        cfg = rest.Bm25Config(k=2.0, b=0.5)
        store = self._store(bm25_config=cfg)
        doc = store._make_bm25_document("test")
        assert doc.options == cfg


# ---------------------------------------------------------------------------
# Point building (indexing)
# ---------------------------------------------------------------------------


class TestBuildPointsNativeBm25:
    """Verify _build_points uses rest.Document for sparse vectors."""

    def _store_and_nodes(self):
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_native_bm25=True,
        )
        nodes = [
            TextNode(
                text="first document",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="src-0")
                },
            ),
            TextNode(
                text="second document",
                id_="22222222-2222-2222-2222-222222222222",
                embedding=[0.0, 1.0],
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="src-1")
                },
            ),
        ]
        return store, nodes

    def test_points_contain_document_objects(self):
        store, nodes = self._store_and_nodes()
        points, ids = store._build_points(nodes, store.sparse_vector_name)

        assert len(points) == 2
        assert len(ids) == 2

        for point in points:
            vector_dict = point.vector
            # Dense vector should be a plain list of floats
            assert isinstance(vector_dict[DEFAULT_DENSE_VECTOR_NAME], list)
            # Sparse vector should be a Document, not a SparseVector
            sparse_val = vector_dict[DEFAULT_SPARSE_VECTOR_NAME]
            assert isinstance(sparse_val, rest.Document)
            assert sparse_val.model == DEFAULT_NATIVE_BM25_MODEL

    def test_document_text_matches_node_content(self):
        store, nodes = self._store_and_nodes()
        points, _ = store._build_points(nodes, store.sparse_vector_name)

        # The Document text should come from the node's embed content
        assert points[0].vector[DEFAULT_SPARSE_VECTOR_NAME].text == "first document"
        assert points[1].vector[DEFAULT_SPARSE_VECTOR_NAME].text == "second document"


# ---------------------------------------------------------------------------
# Collection creation
# ---------------------------------------------------------------------------


class TestCollectionCreationNativeBm25:
    """Verify _create_collection uses IDF modifier for native BM25."""

    def test_sparse_config_uses_idf_modifier(self):
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_native_bm25=True,
        )

        store._create_collection("test", vector_size=2)

        # The create_collection call should include sparse_vectors_config
        call_kwargs = client.create_collection.call_args
        sparse_cfg = call_kwargs.kwargs.get("sparse_vectors_config", {})
        assert store.sparse_vector_name in sparse_cfg
        params = sparse_cfg[store.sparse_vector_name]
        assert params.modifier == rest.Modifier.IDF

    def test_user_sparse_config_overrides_default(self):
        """If the user provides explicit sparse_config, it should be used as-is."""
        custom_config = rest.SparseVectorParams(
            index=rest.SparseIndexParams(on_disk=True),
            modifier=rest.Modifier.NONE,
        )
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_native_bm25=True,
            sparse_config=custom_config,
        )

        store._create_collection("test", vector_size=2)

        call_kwargs = client.create_collection.call_args
        sparse_cfg = call_kwargs.kwargs.get("sparse_vectors_config", {})
        params = sparse_cfg[store.sparse_vector_name]
        assert params.modifier == rest.Modifier.NONE
        assert params.index.on_disk is True


# ---------------------------------------------------------------------------
# Query dispatch
# ---------------------------------------------------------------------------


class TestQueryNativeBm25:
    """Verify query() and aquery() use Document objects for sparse queries."""

    def _make_store(self):
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = True

        # Simulate collection with named dense + sparse vectors
        class _Params:
            vectors = {DEFAULT_DENSE_VECTOR_NAME: object()}
            sparse_vectors = {DEFAULT_SPARSE_VECTOR_NAME: object()}

        class _Config:
            params = _Params()

        class _Collection:
            config = _Config()

        client.get_collection.return_value = _Collection()

        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_native_bm25=True,
        )
        return store, client

    def _fake_scored_point(self, point_id, score, text="dummy"):
        return MagicMock(
            id=point_id,
            score=score,
            vector={DEFAULT_DENSE_VECTOR_NAME: [1.0, 0.0]},
            payload={"text": text, "_node_content": None},
        )

    def test_hybrid_query_sends_document(self):
        store, client = self._make_store()

        # Mock the batch response: [dense_results, sparse_results]
        dense_point = self._fake_scored_point("id-1", 0.9, "doc1")
        sparse_point = self._fake_scored_point("id-2", 0.8, "doc2")
        client.query_batch_points.return_value = [
            MagicMock(points=[dense_point]),
            MagicMock(points=[sparse_point]),
        ]

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="search text",
            similarity_top_k=2,
            sparse_top_k=2,
            hybrid_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
        )
        store.query(query)

        # Verify the batch request was made
        call_args = client.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 2

        # First request is the dense query (plain embedding)
        assert requests[0].using == DEFAULT_DENSE_VECTOR_NAME

        # Second request should use a Document object for the sparse query
        sparse_req = requests[1]
        assert sparse_req.using == DEFAULT_SPARSE_VECTOR_NAME
        assert isinstance(sparse_req.query, rest.Document)
        assert sparse_req.query.text == "search text"
        assert sparse_req.query.model == DEFAULT_NATIVE_BM25_MODEL

    def test_sparse_only_query_sends_document(self):
        store, client = self._make_store()

        sparse_point = self._fake_scored_point("id-1", 0.9, "doc1")
        client.query_batch_points.return_value = [
            MagicMock(points=[sparse_point]),
        ]

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="keyword search",
            similarity_top_k=2,
            mode=VectorStoreQueryMode.SPARSE,
        )
        store.query(query)

        call_args = client.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 1
        assert isinstance(requests[0].query, rest.Document)
        assert requests[0].query.text == "keyword search"

    def test_default_mode_skips_sparse(self):
        """DEFAULT mode should do dense-only search even with native BM25 enabled."""
        store, client = self._make_store()

        dense_point = self._fake_scored_point("id-1", 0.9)
        client.query_batch_points.return_value = [
            MagicMock(points=[dense_point]),
        ]

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            similarity_top_k=1,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        store.query(query)

        # Should use query_batch_points (hybrid-enabled path) but only dense
        call_args = client.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 1
        assert requests[0].using == DEFAULT_DENSE_VECTOR_NAME

    def test_hybrid_query_without_query_str_falls_back_to_dense(self):
        """If query_str is None, hybrid should degrade to dense-only."""
        store, client = self._make_store()

        dense_point = self._fake_scored_point("id-1", 0.9)
        client.query_batch_points.return_value = [
            MagicMock(points=[dense_point]),
        ]

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str=None,
            similarity_top_k=1,
            mode=VectorStoreQueryMode.HYBRID,
        )
        # With no query_str, can_sparse_search will be False,
        # so it falls through to the enable_hybrid dense-only branch
        store.query(query)

        call_args = client.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 1


# ---------------------------------------------------------------------------
# Async query dispatch
# ---------------------------------------------------------------------------


class TestAsyncQueryNativeBm25:
    """Verify aquery() mirrors the sync query logic for native BM25."""

    def _make_store(self):
        client = MagicMock(spec=QdrantClient)
        aclient = MagicMock(spec=AsyncQdrantClient)
        client.collection_exists.return_value = True

        class _Params:
            vectors = {DEFAULT_DENSE_VECTOR_NAME: object()}
            sparse_vectors = {DEFAULT_SPARSE_VECTOR_NAME: object()}

        class _Config:
            params = _Params()

        class _Collection:
            config = _Config()

        client.get_collection.return_value = _Collection()

        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            aclient=aclient,
            enable_native_bm25=True,
        )
        return store, aclient

    def _fake_scored_point(self, point_id, score, text="dummy"):
        return MagicMock(
            id=point_id,
            score=score,
            vector={DEFAULT_DENSE_VECTOR_NAME: [1.0, 0.0]},
            payload={"text": text, "_node_content": None},
        )

    @pytest.mark.asyncio
    async def test_async_hybrid_query_sends_document(self):
        store, aclient = self._make_store()

        dense_point = self._fake_scored_point("id-1", 0.9, "doc1")
        sparse_point = self._fake_scored_point("id-2", 0.8, "doc2")
        aclient.query_batch_points = AsyncMock(
            return_value=[
                MagicMock(points=[dense_point]),
                MagicMock(points=[sparse_point]),
            ]
        )

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="async search",
            similarity_top_k=2,
            sparse_top_k=2,
            hybrid_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
        )
        await store.aquery(query)

        call_args = aclient.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 2

        sparse_req = requests[1]
        assert isinstance(sparse_req.query, rest.Document)
        assert sparse_req.query.text == "async search"

    @pytest.mark.asyncio
    async def test_async_sparse_query_sends_document(self):
        store, aclient = self._make_store()

        sparse_point = self._fake_scored_point("id-1", 0.9)
        aclient.query_batch_points = AsyncMock(
            return_value=[MagicMock(points=[sparse_point])]
        )

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            query_str="async keyword",
            similarity_top_k=2,
            mode=VectorStoreQueryMode.SPARSE,
        )
        await store.aquery(query)

        call_args = aclient.query_batch_points.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 1
        assert isinstance(requests[0].query, rest.Document)


# ---------------------------------------------------------------------------
# Backward compatibility: existing paths still work
# ---------------------------------------------------------------------------


class TestExistingHybridUnchanged:
    """Confirm that the classic FastEmbed/custom encoder path is unaffected."""

    def test_fastembed_hybrid_still_uses_sparse_vector(self):
        """When native BM25 is off, _build_points should produce SparseVector."""
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False

        # Provide a trivial sparse encoder that returns fixed values
        def fake_sparse_fn(texts):
            indices = [[0, 1]] * len(texts)
            values = [[0.5, 0.3]] * len(texts)
            return indices, values

        store = QdrantVectorStore(
            collection_name="test",
            client=client,
            enable_hybrid=True,
            sparse_doc_fn=fake_sparse_fn,
            sparse_query_fn=fake_sparse_fn,
        )

        nodes = [
            TextNode(
                text="test",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
            ),
        ]

        points, _ = store._build_points(nodes, store.sparse_vector_name)

        sparse_val = points[0].vector[DEFAULT_SPARSE_VECTOR_NAME]
        assert isinstance(sparse_val, rest.SparseVector)
        assert sparse_val.indices == [0, 1]
        assert sparse_val.values == [0.5, 0.3]

    def test_non_hybrid_store_unchanged(self):
        """A plain dense-only store should work exactly as before."""
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False

        store = QdrantVectorStore(
            collection_name="test",
            client=client,
        )

        assert store.enable_hybrid is False
        assert store.enable_native_bm25 is False

        nodes = [
            TextNode(
                text="test",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
            ),
        ]
        points, _ = store._build_points(nodes, store.sparse_vector_name)

        # Should only have dense vector, no sparse key
        assert DEFAULT_SPARSE_VECTOR_NAME not in points[0].vector
        assert DEFAULT_DENSE_VECTOR_NAME in points[0].vector
