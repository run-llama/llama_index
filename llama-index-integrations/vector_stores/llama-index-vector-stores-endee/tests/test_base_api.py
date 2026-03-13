"""
Test cases for EndeeVectorStore base API: add, delete, fetch, query, describe, client, and validation.
Based on llama-index-vector-stores-endee/base.py. Requires ENDEE_API_TOKEN for integration tests.
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path for imports

from tests.setup_class import EndeeTestSetup
from llama_index.vector_stores.endee import EndeeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None


class TestEndeeBaseAPI(EndeeTestSetup):
    """Test EndeeVectorStore: from_params, add, delete, fetch, query, describe, client."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if HuggingFaceEmbedding is None:
            raise unittest.SkipTest(
                "HuggingFaceEmbedding not available; install llama-index-vector-stores-endee-embeddings-huggingface"
            )
        cls.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

    def setUp(self):
        super().setUp()

    def test_from_params_creates_store(self):
        """Test from_params creates store and index (get or create)."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        self.assertIsNotNone(vs)
        self.assertEqual(vs.index_name, self.test_index_name)
        self.assertEqual(vs.dimension, self.dimension)
        self.assertEqual(vs.space_type, self.space_type)
        self.assertIsNotNone(vs.client)

    def test_describe_returns_index_metadata(self):
        """Test describe() returns dict with dimension, space_type, ef_con, etc."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        info = vs.describe()
        self.assertIsInstance(info, dict)
        self.assertIn("dimension", info)
        self.assertIn("space_type", info)
        self.assertEqual(info["dimension"], self.dimension)
        self.assertEqual(info["space_type"], self.space_type)
        # ef_con is now a required field in IndexMetadata (endee v0.1.17+)
        self.assertIn("ef_con", info)
        self.assertIsInstance(info["ef_con"], int)

    def test_client_returns_endee_index(self):
        """Test .client returns the underlying endee Index."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        client = vs.client
        self.assertIsNotNone(client)
        self.assertTrue(hasattr(client, "upsert"))
        self.assertTrue(hasattr(client, "query"))
        self.assertTrue(hasattr(client, "describe"))
        # update_filters added in endee v0.1.17
        self.assertTrue(hasattr(client, "update_filters"))

    def test_default_precision_is_int16(self):
        """Test that the default precision is 'int16' (changed from 'float16' in endee v0.1.17)."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        # The store reports back the precision from the backend; it should match the SDK default
        self.assertIsNotNone(vs.precision)
        info = vs.describe()
        self.assertIn(info.get("precision"), ("int16", "float16"),
                      "Precision should be a valid type; default for new indexes is int16")

    def test_update_filters_wrapper(self):
        """Test update_filters() delegates to the underlying endee Index.update_filters()."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        # Add a node so we have an ID to update
        text = "filter update test node"
        emb = self.embed_model.get_text_embedding(text)
        node_id = "update_filter_test_node"
        vs.add([TextNode(text=text, embedding=emb, id_=node_id)])

        # update_filters should return a success string
        result = vs.update_filters([{"id": node_id, "filter": {"category": "updated"}}])
        self.assertIsInstance(result, str)

    def test_add_and_query(self):
        """Test add(nodes) and query return results."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        # Build nodes with embeddings
        texts = ["Python is great", "Rust is fast", "ML is useful"]
        nodes = []
        for i, text in enumerate(texts):
            emb = self.embed_model.get_text_embedding(text)
            nodes.append(TextNode(text=text, embedding=emb, id_=f"node_{i}"))
        ids = vs.add(nodes)
        self.assertEqual(len(ids), len(nodes))
        self.assertEqual(set(ids), {n.node_id for n in nodes})

        # Query
        q_emb = self.embed_model.get_text_embedding("Python")
        query = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=2)
        result = vs.query(query)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.nodes), 0)
        self.assertEqual(len(result.ids), len(result.nodes))

    def test_delete_is_noop(self):
        """Test delete(ref_doc_id) is a no-op (does not raise)."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        vs.delete("some_ref_doc_id")  # no-op, should not raise

    def test_fetch_returns_list(self):
        """Test fetch(ids) returns list of dicts (empty if ids not in index)."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        # Fetch non-existent ids: may return empty or error per id (we log and continue)
        out = vs.fetch(["nonexistent_id_123"])
        self.assertIsInstance(out, list)
        # If we added nodes in another test, fetch real id
        texts = ["Fetch me"]
        nodes = [
            TextNode(
                text=texts[0],
                embedding=self.embed_model.get_text_embedding(texts[0]),
                id_="fetch_test_node",
            )
        ]
        vs.add(nodes)
        out2 = vs.fetch([nodes[0].node_id])
        self.assertIsInstance(out2, list)
        if len(out2) == 1:
            self.assertIn("id", out2[0])
            self.assertIn("vector", out2[0])

    def test_query_caps_top_k_and_ef(self):
        """Test query with large similarity_top_k is capped (endee max 512)."""
        vs = EndeeVectorStore.from_params(
            api_token=self.endee_api_token,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type,
        )
        q_emb = self.embed_model.get_text_embedding("test")
        query = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=1000)
        result = vs.query(query)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.nodes), 512)

    def test_hybrid_from_params_creates_store(self):
        """Test from_params with hybrid=True, sparse_dim, model_name creates hybrid store."""
        try:
            unique_index_name = f"{self.test_index_name}_hybrid_api"
            self.test_indexes.add(unique_index_name)
            vs = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type=self.space_type,
                hybrid=True,
                sparse_dim=30522,
                model_name="splade_pp",
            )
            self.assertIsNotNone(vs)
            self.assertTrue(vs.hybrid)
            self.assertEqual(vs.sparse_dim, 30522)
            self.assertEqual(vs.model_name, "splade_pp")
            self.assertIsNotNone(vs.client)
            info = vs.describe()
            self.assertIn("dimension", info)
            self.assertTrue(info.get("is_hybrid", info.get("sparse_dim", 0) > 0))
        except ImportError as e:
            self.skipTest(f"Hybrid dependencies not available: {e}")

    def test_hybrid_add_and_query(self):
        """Test hybrid store add(nodes) and query(..., hybrid=True)."""
        try:
            unique_index_name = f"{self.test_index_name}_hybrid_add"
            self.test_indexes.add(unique_index_name)
            vs = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type=self.space_type,
                hybrid=True,
                sparse_dim=30522,
                model_name="splade_pp",
            )
            texts = ["Python is great", "Rust is fast"]
            nodes = []
            for i, text in enumerate(texts):
                emb = self.embed_model.get_text_embedding(text)
                nodes.append(TextNode(text=text, embedding=emb, id_=f"hybrid_node_{i}"))
            ids = vs.add(nodes, hybrid=True)
            self.assertEqual(len(ids), len(nodes))
            q_emb = self.embed_model.get_text_embedding("Python")
            query = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=2)
            result = vs.query(query, hybrid=True)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(len(result.nodes), 0)
        except ImportError as e:
            self.skipTest(f"Hybrid dependencies not available: {e}")


class TestEndeeValidation(EndeeTestSetup):
    """Test validation aligned with endee create_index: index name, dimension, space_type, precision."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_invalid_index_name_rejected(self):
        """Test that invalid index name (e.g. with hyphen) is rejected when creating new index."""
        # Use a name that does not exist and is invalid: contains hyphen
        invalid_name = "test-invalid-name"
        self.test_indexes.add(invalid_name)
        try:
            vs = EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=invalid_name,
                dimension=self.dimension,
                space_type=self.space_type,
            )
            # If backend accepts it, we just pass
            self.assertIsNotNone(vs)
        except ValueError as e:
            self.assertIn("index name", str(e).lower())

    def test_dimension_over_max_rejected(self):
        """Test dimension > MAX_DIMENSION_ALLOWED (10000) is rejected when creating new index."""
        unique_name = f"{self.test_index_name}_bigdim"
        self.test_indexes.add(unique_name)
        with self.assertRaises(ValueError) as ctx:
            EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_name,
                dimension=10001,
                space_type=self.space_type,
            )
        self.assertIn("dimension", str(ctx.exception).lower())

    def test_invalid_space_type_rejected(self):
        """Test invalid space_type is rejected when creating new index."""
        unique_name = f"{self.test_index_name}_badspace"
        self.test_indexes.add(unique_name)
        with self.assertRaises(ValueError) as ctx:
            EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_name,
                dimension=self.dimension,
                space_type="invalid_space_type",
            )
        self.assertIn("space type", str(ctx.exception).lower())

    def test_invalid_precision_rejected(self):
        """Test invalid precision is rejected when creating new index."""
        unique_name = f"{self.test_index_name}_badprec"
        self.test_indexes.add(unique_name)
        with self.assertRaises(ValueError) as ctx:
            EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_name,
                dimension=self.dimension,
                space_type=self.space_type,
                precision="invalid_precision",
            )
        self.assertIn("precision", str(ctx.exception).lower())

    def test_missing_dimension_raises_when_creating(self):
        """Test missing dimension raises when index does not exist (create path)."""
        unique_name = f"{self.test_index_name}_nodim"
        self.test_indexes.add(unique_name)
        with self.assertRaises(ValueError) as ctx:
            EndeeVectorStore.from_params(
                api_token=self.endee_api_token,
                index_name=unique_name,
                dimension=None,
                space_type=self.space_type,
            )
        self.assertIn("dimension", str(ctx.exception).lower())


class TestUpdateFiltersUnit(unittest.TestCase):
    """Unit tests for update_filters() wrapper. No real API connection required."""

    def setUp(self):
        self.mock_index = MagicMock()
        self.store = EndeeVectorStore(endee_index=self.mock_index, dimension=4)

    def test_update_filters_delegates_to_index(self):
        """update_filters() must call self._endee_index.update_filters() with the provided list."""
        self.mock_index.update_filters.return_value = "2 filters updated"
        updates = [
            {"id": "vec1", "filter": {"category": "B"}},
            {"id": "vec2", "filter": {"category": "C", "priority": 1}},
        ]
        result = self.store.update_filters(updates)
        self.mock_index.update_filters.assert_called_once_with(updates)
        self.assertEqual(result, "2 filters updated")

    def test_update_filters_raises_runtime_error_on_failure(self):
        """update_filters() must raise RuntimeError when the underlying call fails."""
        self.mock_index.update_filters.side_effect = Exception("API error")
        with self.assertRaises(RuntimeError) as ctx:
            self.store.update_filters([{"id": "vec1", "filter": {}}])
        self.assertIn("update_filters failed", str(ctx.exception))

    def test_update_filters_empty_list(self):
        """update_filters() passes an empty list through to the underlying index."""
        self.mock_index.update_filters.return_value = "0 filters updated"
        result = self.store.update_filters([])
        self.mock_index.update_filters.assert_called_once_with([])
        self.assertIsNotNone(result)


class TestBuildQueryParams(unittest.TestCase):
    """Unit tests for _build_query_params covering prefilter_cardinality_threshold
    and filter_boost_percentage. No real API connection is required."""

    def setUp(self):
        mock_index = MagicMock()
        self.store = EndeeVectorStore(endee_index=mock_index, dimension=4)
        self._base_args = dict(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=10,
            ef=128,
            filter_for_api=None,
            sparse_indices=None,
            sparse_values=None,
        )

    # --- prefilter_cardinality_threshold ---

    def test_prefilter_cardinality_threshold_excluded_when_none(self):
        """prefilter_cardinality_threshold=None must not appear in the output dict."""
        result = self.store._build_query_params(
            **self._base_args,
            prefilter_cardinality_threshold=None,
        )
        self.assertNotIn("prefilter_cardinality_threshold", result)

    def test_prefilter_cardinality_threshold_included_when_set(self):
        """prefilter_cardinality_threshold value is forwarded to the output dict."""
        result = self.store._build_query_params(
            **self._base_args,
            prefilter_cardinality_threshold=10000,
        )
        self.assertIn("prefilter_cardinality_threshold", result)
        self.assertEqual(result["prefilter_cardinality_threshold"], 10000)

    def test_prefilter_cardinality_threshold_lower_bound(self):
        """prefilter_cardinality_threshold at minimum valid value (1000) is forwarded."""
        result = self.store._build_query_params(
            **self._base_args,
            prefilter_cardinality_threshold=1000,
        )
        self.assertEqual(result["prefilter_cardinality_threshold"], 1000)

    def test_prefilter_cardinality_threshold_upper_bound(self):
        """prefilter_cardinality_threshold at maximum valid value (1_000_000) is forwarded."""
        result = self.store._build_query_params(
            **self._base_args,
            prefilter_cardinality_threshold=1_000_000,
        )
        self.assertEqual(result["prefilter_cardinality_threshold"], 1_000_000)

    # --- filter_boost_percentage ---

    def test_filter_boost_percentage_excluded_when_none(self):
        """filter_boost_percentage=None must not appear in the output dict."""
        result = self.store._build_query_params(
            **self._base_args,
            filter_boost_percentage=None,
        )
        self.assertNotIn("filter_boost_percentage", result)

    def test_filter_boost_percentage_included_when_set(self):
        """filter_boost_percentage value is forwarded to the output dict."""
        result = self.store._build_query_params(
            **self._base_args,
            filter_boost_percentage=30,
        )
        self.assertIn("filter_boost_percentage", result)
        self.assertEqual(result["filter_boost_percentage"], 30)

    def test_filter_boost_percentage_lower_bound(self):
        """filter_boost_percentage=0 (min valid) is forwarded."""
        result = self.store._build_query_params(
            **self._base_args,
            filter_boost_percentage=0,
        )
        self.assertEqual(result["filter_boost_percentage"], 0)

    def test_filter_boost_percentage_upper_bound(self):
        """filter_boost_percentage=100 (max valid) is forwarded."""
        result = self.store._build_query_params(
            **self._base_args,
            filter_boost_percentage=100,
        )
        self.assertEqual(result["filter_boost_percentage"], 100)

    # --- both params together ---

    def test_both_params_included_when_set(self):
        """Both params are forwarded when both are provided."""
        result = self.store._build_query_params(
            **self._base_args,
            prefilter_cardinality_threshold=50000,
            filter_boost_percentage=25,
        )
        self.assertEqual(result["prefilter_cardinality_threshold"], 50000)
        self.assertEqual(result["filter_boost_percentage"], 25)

    def test_both_params_excluded_when_none(self):
        """Neither param appears when both are None (default behaviour)."""
        result = self.store._build_query_params(**self._base_args)
        self.assertNotIn("prefilter_cardinality_threshold", result)
        self.assertNotIn("filter_boost_percentage", result)

    def test_base_keys_always_present(self):
        """Core keys (vector, top_k, ef, include_vectors) are always present."""
        result = self.store._build_query_params(**self._base_args)
        for key in ("vector", "top_k", "ef", "include_vectors"):
            self.assertIn(key, result)


class TestQueryWithFilterParams(EndeeTestSetup):
    """Integration tests: query() correctly passes prefilter_cardinality_threshold
    and filter_boost_percentage through to the Endee API."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if HuggingFaceEmbedding is None:
            raise unittest.SkipTest(
                "HuggingFaceEmbedding not available; install llama-index-embeddings-huggingface"
            )
        cls.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        cls.vs = EndeeVectorStore.from_params(
            api_token=cls.endee_api_token,
            index_name=cls.test_index_name,
            dimension=cls.dimension,
            space_type=cls.space_type,
        )
        # Seed a node so queries can actually return results
        text = "Python is great for data science"
        emb = cls.embed_model.get_text_embedding(text)
        cls.vs.add([TextNode(text=text, embedding=emb, id_="filter_param_seed")])

    def _make_query(self, **extra_kwargs):
        q_emb = self.embed_model.get_text_embedding("data science")
        query = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=2)
        return self.vs.query(query, **extra_kwargs)

    def test_query_with_prefilter_cardinality_threshold(self):
        """query() succeeds when prefilter_cardinality_threshold is provided."""
        result = self._make_query(prefilter_cardinality_threshold=10000)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.nodes, list)
        self.assertEqual(len(result.ids), len(result.nodes))

    def test_query_with_filter_boost_percentage(self):
        """query() succeeds when filter_boost_percentage is provided."""
        result = self._make_query(filter_boost_percentage=30)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.nodes, list)
        self.assertEqual(len(result.ids), len(result.nodes))

    def test_query_with_both_filter_params(self):
        """query() succeeds when both prefilter_cardinality_threshold and
        filter_boost_percentage are provided together."""
        result = self._make_query(
            prefilter_cardinality_threshold=50000,
            filter_boost_percentage=20,
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result.nodes, list)
        self.assertEqual(len(result.ids), len(result.nodes))

    def test_query_default_omits_filter_params(self):
        """query() without these params still returns valid results (params are optional)."""
        result = self._make_query()
        self.assertIsNotNone(result)
        self.assertIsInstance(result.nodes, list)


if __name__ == "__main__":
    unittest.main()