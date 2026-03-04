"""
Unit tests for Azure CosmosDB NoSQL Vector Store.

Tests core methods and helpers without any database access using mocks.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from llama_index.vector_stores.azurecosmosnosql.utils import (
    AzureCosmosDBNoSqlVectorSearchType,
    Constants,
    ParamMapping,
)
from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VECTOR_EMBEDDING_POLICY: Dict[str, Any] = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3,
        }
    ]
}

INDEXING_POLICY: Dict[str, Any] = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

FULL_TEXT_INDEXING_POLICY: Dict[str, Any] = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    "fullTextIndexes": [{"path": "/text"}],
}

COSMOS_CONTAINER_PROPERTIES: Dict[str, Any] = {
    "partition_key": MagicMock()  # PartitionKey mock
}


def _make_store(
    full_text_search_enabled: bool = False,
    indexing_policy: Dict[str, Any] = None,
    container_properties: Dict[str, Any] = None,
) -> AzureCosmosDBNoSqlVectorSearch:
    """Create a store instance backed entirely by mocks (no real DB calls)."""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_container = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_db
    mock_db.create_container_if_not_exists.return_value = mock_container

    store = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=mock_client,
        vector_embedding_policy=VECTOR_EMBEDDING_POLICY,
        indexing_policy=indexing_policy or INDEXING_POLICY,
        cosmos_container_properties=container_properties or COSMOS_CONTAINER_PROPERTIES,
        cosmos_database_properties={},
        database_name="test_db",
        container_name="test_container",
        full_text_search_enabled=full_text_search_enabled,
    )
    return store


# ===========================================================================
# ParamMapping tests
# ===========================================================================

class TestParamMapping:
    def test_add_parameter(self) -> None:
        pm = ParamMapping(table="c")
        pm.add_parameter("limit", 10)
        assert "limit" in pm.parameter_map
        assert pm.parameter_map["limit"].key == "@limit"
        assert pm.parameter_map["limit"].value == 10

    def test_add_parameter_idempotent(self) -> None:
        """add_parameter overwrites an existing key — gen_param_key is idempotent."""
        pm = ParamMapping(table="c")
        pm.add_parameter("limit", 10)
        pm.add_parameter("limit", 99)  # add_parameter always overwrites
        assert pm.parameter_map["limit"].value == 99

    def test_gen_param_key(self) -> None:
        pm = ParamMapping(table="c")
        key = pm.gen_param_key("limit", 5)
        assert key == "@limit"
        assert pm.parameter_map["limit"].value == 5

    def test_gen_param_key_idempotent(self) -> None:
        pm = ParamMapping(table="c")
        pm.gen_param_key("limit", 5)
        key2 = pm.gen_param_key("limit", 99)
        assert key2 == "@limit"
        assert pm.parameter_map["limit"].value == 5  # unchanged

    def test_gen_proj_field_no_alias(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_proj_field(key="textKey", value="text")
        assert result == "c[@textKey]"
        assert "@textKey" in result

    def test_gen_proj_field_with_alias(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_proj_field(key="textKey", value="text", alias="text")
        assert result == "c[@textKey] as text"

    def test_gen_vector_distance_proj_field_no_alias(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_vector_distance_proj_field(
            vector_field="embedding", vector=[1.0, 0.0, 0.0]
        )
        assert result.startswith("VectorDistance(c[@vectorKey], @vector)")

    def test_gen_vector_distance_proj_field_with_alias(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_vector_distance_proj_field(
            vector_field="embedding", vector=[1.0, 0.0, 0.0], alias="SimilarityScore"
        )
        assert "VectorDistance" in result
        assert "as SimilarityScore" in result

    def test_gen_vector_distance_order_by_field_inline_literal(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_vector_distance_order_by_field(
            vector_field="embedding", vector=[1.0, 0.0, 0.0]
        )
        # Must use direct path (c.embedding) and inline array, NOT @vector param
        assert "c.embedding" in result
        assert "[1.0, 0.0, 0.0]" in result
        assert "@vector" not in result

    def test_gen_vector_distance_order_by_field_with_alias(self) -> None:
        pm = ParamMapping(table="c")
        result = pm.gen_vector_distance_order_by_field(
            vector_field="embedding", vector=[0.5, 0.5, 0.0], alias="Score"
        )
        assert "as Score" in result

    def test_export_parameter_list(self) -> None:
        pm = ParamMapping(table="c")
        pm.add_parameter("limit", 5)
        pm.add_parameter("vector", [1.0, 0.0])
        params = pm.export_parameter_list()
        assert len(params) == 2
        names = {p["name"] for p in params}
        assert "@limit" in names
        assert "@vector" in names

    def test_export_parameter_list_custom_keys(self) -> None:
        pm = ParamMapping(table="c", name_key="n", value_key="v")
        pm.add_parameter("limit", 10)
        params = pm.export_parameter_list()
        assert params[0]["n"] == "@limit"
        assert params[0]["v"] == 10


# ===========================================================================
# AzureCosmosDBNoSqlVectorSearchType enum tests
# ===========================================================================

class TestSearchTypeEnum:
    def test_all_values_exist(self) -> None:
        values = {t.value for t in AzureCosmosDBNoSqlVectorSearchType}
        assert "vector" in values
        assert "vector_score_threshold" in values
        assert "full_text_search" in values
        assert "full_text_ranking" in values
        assert "hybrid" in values
        assert "hybrid_score_threshold" in values

    def test_string_comparison(self) -> None:
        assert AzureCosmosDBNoSqlVectorSearchType.VECTOR == "vector"
        assert AzureCosmosDBNoSqlVectorSearchType.HYBRID == "hybrid"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            AzureCosmosDBNoSqlVectorSearchType("invalid_type")


# ===========================================================================
# Store initialisation & property tests
# ===========================================================================

class TestStoreInit:
    def test_embedding_key_extracted_from_policy(self) -> None:
        store = _make_store()
        # path is "/embedding" → key should be "embedding" (leading slash stripped)
        assert store._embedding_key == "embedding"

    def test_default_keys(self) -> None:
        store = _make_store()
        assert store._id_key == "id"
        assert store._text_key == "text"
        assert store._metadata_key == "metadata"
        assert store._table_alias == "c"

    def test_full_text_search_disabled_by_default(self) -> None:
        store = _make_store()
        assert store._full_text_search_enabled is False

    def test_full_text_search_enabled(self) -> None:
        store = _make_store(full_text_search_enabled=True)
        assert store._full_text_search_enabled is True

    def test_missing_vector_indexes_raises(self) -> None:
        bad_policy = {
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [],
            "vectorIndexes": [],
        }
        with pytest.raises(ValueError, match="vectorIndexes"):
            _make_store(indexing_policy=bad_policy)

    def test_missing_partition_key_raises(self) -> None:
        with pytest.raises(ValueError, match="partition_key"):
            _make_store(container_properties={"partition_key": None})

    def test_client_property(self) -> None:
        store = _make_store()
        assert store.client is store._cosmos_client


# ===========================================================================
# _is_* helper method tests
# ===========================================================================

class TestIsHelpers:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)

    def test_is_vector_search_with_threshold_true(self) -> None:
        assert self.store._is_vector_search_with_threshold(
            AzureCosmosDBNoSqlVectorSearchType.VECTOR_SCORE_THRESHOLD
        )

    def test_is_vector_search_with_threshold_false_for_vector(self) -> None:
        assert not self.store._is_vector_search_with_threshold(
            AzureCosmosDBNoSqlVectorSearchType.VECTOR
        )

    def test_is_vector_search_with_threshold_false_for_hybrid(self) -> None:
        # Hybrid is excluded from threshold filtering
        assert not self.store._is_vector_search_with_threshold(
            AzureCosmosDBNoSqlVectorSearchType.HYBRID_SCORE_THRESHOLD
        )

    def test_is_full_text_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH,
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID_SCORE_THRESHOLD,
            AzureCosmosDBNoSqlVectorSearchType.WEIGHTED_HYBRID_SEARCH,
        ):
            assert self.store._is_full_text_search_type(st), f"{st} should be full text"

    def test_is_not_full_text_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.VECTOR,
            AzureCosmosDBNoSqlVectorSearchType.VECTOR_SCORE_THRESHOLD,
        ):
            assert not self.store._is_full_text_search_type(st), f"{st} should not be full text"

    def test_is_vector_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.VECTOR,
            AzureCosmosDBNoSqlVectorSearchType.VECTOR_SCORE_THRESHOLD,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID_SCORE_THRESHOLD,
            AzureCosmosDBNoSqlVectorSearchType.WEIGHTED_HYBRID_SEARCH,
        ):
            assert self.store._is_vector_search_type(st), f"{st} should be vector"

    def test_is_not_vector_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH,
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
        ):
            assert not self.store._is_vector_search_type(st), f"{st} should not be vector"


# ===========================================================================
# _validate_search_args tests
# ===========================================================================

class TestValidateSearchArgs:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)
        self.store_no_fts = _make_store(full_text_search_enabled=False)

    def test_valid_vector_search(self) -> None:
        # Should not raise
        self.store._validate_search_args(
            search_type="vector", vector=[1.0, 0.0, 0.0]
        )

    def test_invalid_search_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid search_type"):
            self.store._validate_search_args(search_type="bad_type")

    def test_vector_search_missing_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="Embedding must be provided"):
            self.store._validate_search_args(search_type="vector", vector=None)

    def test_full_text_search_disabled_raises(self) -> None:
        with pytest.raises(ValueError, match="Full text search is not enabled"):
            self.store_no_fts._validate_search_args(
                search_type="full_text_search"
            )

    def test_full_text_search_enabled_valid(self) -> None:
        # Should not raise
        self.store._validate_search_args(search_type="full_text_search")

    def test_return_with_vectors_on_non_vector_type_raises(self) -> None:
        with pytest.raises(ValueError, match="return_with_vectors"):
            self.store._validate_search_args(
                search_type="full_text_search", return_with_vectors=True
            )

    def test_return_with_vectors_on_vector_type_valid(self) -> None:
        # Should not raise
        self.store._validate_search_args(
            search_type="vector", vector=[1.0, 0.0], return_with_vectors=True
        )

    def test_hybrid_search_requires_vector(self) -> None:
        with pytest.raises(ValueError, match="Embedding must be provided"):
            self.store._validate_search_args(search_type="hybrid", vector=None)


# ===========================================================================
# _generate_limit_clause tests
# ===========================================================================

class TestGenerateLimitClause:
    def setup_method(self) -> None:
        self.store = _make_store()

    def test_limit_clause(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_limit_clause(pm, limit=5)
        assert result == " TOP @limit"
        # Parameter should be registered
        assert pm.parameter_map["limit"].value == 5

    def test_limit_clause_different_values(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_limit_clause(pm, limit=10)
        assert "@limit" in result
        assert pm.parameter_map["limit"].value == 10


# ===========================================================================
# _generate_order_by_component_with_full_text_rank_filter tests
# ===========================================================================

class TestGenerateOrderByComponent:
    def setup_method(self) -> None:
        self.store = _make_store()

    def test_single_term(self) -> None:
        result = self.store._generate_order_by_component_with_full_text_rank_filter(
            {"search_field": "text", "search_text": "lorem"}
        )
        assert result == "FullTextScore(c.text, 'lorem')"

    def test_multiple_terms(self) -> None:
        result = self.store._generate_order_by_component_with_full_text_rank_filter(
            {"search_field": "text", "search_text": "lorem ipsum"}
        )
        assert result == "FullTextScore(c.text, 'lorem', 'ipsum')"

    def test_different_field(self) -> None:
        result = self.store._generate_order_by_component_with_full_text_rank_filter(
            {"search_field": "description", "search_text": "quick brown"}
        )
        assert "c.description" in result
        assert "'quick'" in result
        assert "'brown'" in result

    def test_uses_direct_path_not_bracket(self) -> None:
        result = self.store._generate_order_by_component_with_full_text_rank_filter(
            {"search_field": "text", "search_text": "foo"}
        )
        # Must NOT use bracket indexer c[@...], must use c.text
        assert "c[@" not in result
        assert "c.text" in result


# ===========================================================================
# _generate_order_by_clause tests
# ===========================================================================

class TestGenerateOrderByClause:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)

    def test_vector_search(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="vector", param_mapping=pm, vector=[1.0, 0.0, 0.0]
        )
        assert "ORDER BY" in result
        assert "VectorDistance" in result
        assert "@vector" in result

    def test_vector_score_threshold(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="vector_score_threshold", param_mapping=pm, vector=[0.5, 0.5, 0.0]
        )
        assert "ORDER BY" in result
        assert "VectorDistance" in result

    def test_full_text_ranking_single_filter(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="full_text_ranking",
            param_mapping=pm,
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        # Single component → ORDER BY RANK <component> (no RRF)
        assert "ORDER BY RANK" in result
        assert "FullTextScore" in result
        assert "RRF" not in result

    def test_full_text_ranking_multiple_filters_uses_rrf(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="full_text_ranking",
            param_mapping=pm,
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem"},
                {"search_field": "text", "search_text": "ipsum"},
            ],
        )
        assert "ORDER BY RANK RRF(" in result
        assert result.count("FullTextScore") == 2

    def test_hybrid_uses_rrf_with_vector_distance(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "ORDER BY RANK RRF(" in result
        assert "FullTextScore" in result
        assert "VectorDistance" in result
        # Vector must be inline literal, not @vector param
        assert "@vector" not in result
        assert "[1.0, 0.0, 0.0]" in result

    def test_hybrid_score_threshold_uses_rrf(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid_score_threshold",
            param_mapping=pm,
            vector=[0.0, 1.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "foo"}],
        )
        assert "ORDER BY RANK RRF(" in result
        assert "VectorDistance" in result

    def test_full_text_ranking_missing_filter_raises(self) -> None:
        pm = ParamMapping(table="c")
        with pytest.raises(ValueError, match="full_text_rank_filter"):
            self.store._generate_order_by_clause(
                search_type="full_text_ranking",
                param_mapping=pm,
                full_text_rank_filter=None,
            )

    def test_full_text_search_returns_empty(self) -> None:
        # full_text_search has no ORDER BY clause
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="full_text_search", param_mapping=pm
        )
        assert result.strip() == ""

    def test_weighted_hybrid_emits_weight_clause(self) -> None:
        """WEIGHTED_HYBRID_SEARCH uses the same RRF query as hybrid; weights are client-side."""
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="weighted_hybrid_search",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "ORDER BY RANK RRF(" in result
        assert "FullTextScore" in result
        assert "VectorDistance" in result
        # Weights are applied client-side — must NOT appear in the SQL clause
        assert "weight=" not in result
        assert "weights=" not in result
        # Vector must still be inlined
        assert "[1.0, 0.0, 0.0]" in result
        assert "@vector" not in result

    def test_weighted_hybrid_without_weights_omits_weight_clause(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="weighted_hybrid_search",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=None,
        )
        assert "ORDER BY RANK RRF(" in result
        assert "weight=" not in result

    def test_weighted_hybrid_multiple_text_components(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="weighted_hybrid_search",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem"},
                {"search_field": "text", "search_text": "ipsum"},
            ],
            weights=[0.2, 0.3, 0.5],
        )
        assert "ORDER BY RANK RRF(" in result
        assert result.count("FullTextScore") == 2
        # Weights are client-side — not in SQL
        assert "weight=" not in result


# ===========================================================================
# _generate_projection_fields tests
# ===========================================================================

class TestGenerateProjectionFields:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)

    def test_vector_projection_includes_similarity_score(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="vector", param_mapping=pm, vector=[1.0, 0.0, 0.0]
        )
        assert "SimilarityScore" in result
        assert "VectorDistance" in result

    def test_vector_score_threshold_includes_similarity_score(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="vector_score_threshold",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
        )
        assert "SimilarityScore" in result

    def test_hybrid_projection_excludes_similarity_score(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        # CosmosDB rejects VectorDistance in SELECT with ORDER BY RANK
        assert "SimilarityScore" not in result
        assert "VectorDistance" not in result

    def test_hybrid_uses_direct_path_for_text_and_metadata(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "foo"}],
        )
        # Must use c.text not c[@textKey]
        assert "c.text" in result
        assert "c.metadata" in result
        assert "c[@" not in result

    def test_full_text_ranking_includes_id_text_metadata(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="full_text_ranking",
            param_mapping=pm,
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "c.id" in result
        assert "c.text" in result
        assert "c.metadata" in result

    def test_projection_mapping_overrides_default(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="vector",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            projection_mapping={"id": "id", "text": "content"},
        )
        assert "c.id as id" in result
        assert "c.text as content" in result
        # Vector search still appends SimilarityScore even with projection_mapping
        assert "SimilarityScore" in result

    def test_return_with_vectors_adds_embedding_field(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="vector",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            return_with_vectors=True,
        )
        assert "as vector" in result

    def test_full_text_search_uses_bracket_indexer(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="full_text_search", param_mapping=pm
        )
        # gen_proj_field registers the value as the param key, so key="text" → @text
        assert "c[@text] as text" in result
        assert "c[@metadata] as metadata" in result


# ===========================================================================
# _construct_search_query tests
# ===========================================================================

class TestConstructSearchQuery:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)

    def test_vector_query_structure(self) -> None:
        query, params = self.store._construct_search_query(
            limit=5, search_type="vector", vector=[1.0, 0.0, 0.0]
        )
        assert query.startswith("SELECT TOP @limit")
        assert "FROM c" in query
        assert "ORDER BY VectorDistance" in query
        assert "SimilarityScore" in query
        # @limit and @vector params should be exported
        param_names = {p["name"] for p in params}
        assert "@limit" in param_names
        assert "@vector" in param_names

    def test_vector_score_threshold_query(self) -> None:
        query, params = self.store._construct_search_query(
            limit=3, search_type="vector_score_threshold", vector=[0.5, 0.5, 0.0]
        )
        assert "TOP @limit" in query
        assert "SimilarityScore" in query
        assert "ORDER BY VectorDistance" in query

    def test_full_text_search_query_no_order_by(self) -> None:
        query, params = self.store._construct_search_query(
            limit=5,
            search_type="full_text_search",
            where="FullTextContains(c.text, 'lorem')",
        )
        assert "ORDER BY" not in query
        assert "WHERE FullTextContains" in query
        assert "TOP @limit" in query

    def test_full_text_ranking_query(self) -> None:
        query, params = self.store._construct_search_query(
            limit=5,
            search_type="full_text_ranking",
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "ORDER BY RANK" in query
        assert "FullTextScore" in query
        assert "TOP @limit" in query

    def test_hybrid_query_uses_offset_limit_not_top(self) -> None:
        query, params = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "TOP" not in query
        assert "OFFSET 0 LIMIT 3" in query
        assert "ORDER BY RANK RRF(" in query
        assert "VectorDistance(c.embedding, [1.0, 0.0, 0.0])" in query

    def test_hybrid_score_threshold_query(self) -> None:
        query, params = self.store._construct_search_query(
            limit=5,
            search_type="hybrid_score_threshold",
            vector=[0.0, 1.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "ipsum"}],
        )
        assert "TOP" not in query
        assert "OFFSET 0 LIMIT 5" in query
        assert "ORDER BY RANK RRF(" in query

    def test_where_clause_included(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=5,
            search_type="vector",
            vector=[1.0, 0.0, 0.0],
            where="c.metadata.author = 'King'",
        )
        assert "WHERE c.metadata.author = 'King'" in query

    def test_offset_limit_overrides_top(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=5,
            search_type="vector",
            vector=[1.0, 0.0, 0.0],
            offset_limit="OFFSET 2 LIMIT 3",
        )
        assert "TOP" not in query
        assert "OFFSET 2 LIMIT 3" in query

    def test_projection_mapping_in_query(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="vector",
            vector=[1.0, 0.0, 0.0],
            projection_mapping={"id": "id", "text": "body"},
        )
        assert "c.id as id" in query
        assert "c.text as body" in query

    def test_return_with_vectors_in_query(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="vector",
            vector=[1.0, 0.0, 0.0],
            return_with_vectors=True,
        )
        assert "as vector" in query

    def test_full_text_ranking_multiple_filters_rrf(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="full_text_ranking",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem"},
                {"search_field": "text", "search_text": "ipsum"},
            ],
        )
        assert "ORDER BY RANK RRF(" in query
        assert query.count("FullTextScore") == 2

    def test_parameters_not_empty_for_vector(self) -> None:
        _, params = self.store._construct_search_query(
            limit=5, search_type="vector", vector=[1.0, 0.0, 0.0]
        )
        assert len(params) > 0

    def test_weighted_hybrid_uses_offset_limit_not_top(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="weighted_hybrid_search",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "TOP" not in query
        assert "OFFSET 0 LIMIT 3" in query

    def test_weighted_hybrid_emits_rrf_without_weight_in_sql(self) -> None:
        """Weights are applied client-side; the SQL query is identical to hybrid."""
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="weighted_hybrid_search",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "ORDER BY RANK RRF(" in query
        assert "weight=" not in query
        assert "weights=" not in query
        assert "FullTextScore" in query
        assert "VectorDistance(c.embedding, [1.0, 0.0, 0.0])" in query

    def test_weighted_hybrid_no_weights_same_as_with_weights(self) -> None:
        """SQL output is identical whether weights are provided or not."""
        q_with, _ = self.store._construct_search_query(
            limit=3,
            search_type="weighted_hybrid_search",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        q_without, _ = self.store._construct_search_query(
            limit=3,
            search_type="weighted_hybrid_search",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert q_with == q_without

    def test_hybrid_vector_inline_not_parameterized(self) -> None:
        query, params = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[0.3, 0.7, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "test"}],
        )
        # @vector must NOT appear in params for hybrid (it's inlined)
        param_names = {p["name"] for p in params}
        assert "@vector" not in param_names
        # The inline literal must be in the query
        assert "[0.3, 0.7, 0.0]" in query


# ===========================================================================
# _search_query backward-compat (pre_filter) tests
# ===========================================================================

class TestSearchQueryPreFilterCompat:
    """Verify _search_query is a full backward-compatible replacement for _query.

    The old _query accepted:
        pre_filter = {
            "where_clause":       "WHERE ..."    → maps to `where`
            "limit_offset_clause": "OFFSET x LIMIT y" → maps to `offset_limit`
        }
    It also:
    - defaulted to vector search
    - used TOP k when no limit_offset_clause
    - omitted TOP when limit_offset_clause was given
    - projected c.id, c.text, c.metadata, VectorDistance(...) AS SimilarityScore
    - reconstructed nodes via metadata_dict_to_node
    """

    def setup_method(self) -> None:
        self.store = _make_store()
        from llama_index.core.vector_stores.types import VectorStoreQueryResult
        self._empty_result = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        self.store._execute_search_query = MagicMock(return_value=self._empty_result)

    # ------------------------------------------------------------------
    # pre_filter["where_clause"] → where
    # ------------------------------------------------------------------

    def test_pre_filter_where_clause_mapped(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            pre_filter={"where_clause": "WHERE c.author = 'King'"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "WHERE c.author = 'King'" in constructed_query

    def test_pre_filter_where_clause_with_top_still_present(self) -> None:
        """_query added TOP k when no limit_offset_clause — same in _search_query."""
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            pre_filter={"where_clause": "WHERE c.year = 2020"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "TOP @limit" in constructed_query
        assert "WHERE c.year = 2020" in constructed_query

    # ------------------------------------------------------------------
    # pre_filter["limit_offset_clause"] → offset_limit (replaces TOP)
    # ------------------------------------------------------------------

    def test_pre_filter_limit_offset_clause_mapped(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            pre_filter={"limit_offset_clause": "OFFSET 2 LIMIT 3"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "OFFSET 2 LIMIT 3" in constructed_query

    def test_pre_filter_limit_offset_replaces_top(self) -> None:
        """_query omitted TOP when limit_offset_clause was provided."""
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            pre_filter={"limit_offset_clause": "OFFSET 0 LIMIT 5"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "TOP" not in constructed_query
        assert "OFFSET 0 LIMIT 5" in constructed_query

    # ------------------------------------------------------------------
    # Both clauses together
    # ------------------------------------------------------------------

    def test_pre_filter_both_clauses_together(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            pre_filter={
                "where_clause": "WHERE c.genre = 'horror'",
                "limit_offset_clause": "OFFSET 1 LIMIT 4",
            },
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "WHERE c.genre = 'horror'" in constructed_query
        assert "OFFSET 1 LIMIT 4" in constructed_query
        assert "TOP" not in constructed_query

    # ------------------------------------------------------------------
    # Explicit args win over pre_filter
    # ------------------------------------------------------------------

    def test_explicit_where_takes_priority_over_pre_filter(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            where="WHERE c.year = 2024",
            pre_filter={"where_clause": "WHERE c.author = 'King'"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "c.year = 2024" in constructed_query
        assert "c.author" not in constructed_query

    def test_explicit_offset_limit_takes_priority_over_pre_filter(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
            offset_limit="OFFSET 10 LIMIT 2",
            pre_filter={"limit_offset_clause": "OFFSET 0 LIMIT 99"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "OFFSET 10 LIMIT 2" in constructed_query
        assert "OFFSET 0 LIMIT 99" not in constructed_query

    # ------------------------------------------------------------------
    # Edge cases: None, empty dict, missing keys
    # ------------------------------------------------------------------

    def test_pre_filter_none_is_safe(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=3,
            search_type="vector",
            pre_filter=None,
        )
        assert self.store._execute_search_query.called

    def test_pre_filter_empty_dict_is_safe(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=3,
            search_type="vector",
            pre_filter={},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        # No WHERE injected, TOP still present
        assert "TOP @limit" in constructed_query
        assert "WHERE" not in constructed_query

    def test_pre_filter_missing_where_key_is_safe(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=3,
            search_type="vector",
            pre_filter={"limit_offset_clause": "OFFSET 0 LIMIT 3"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "WHERE" not in constructed_query

    def test_pre_filter_missing_limit_offset_key_is_safe(self) -> None:
        self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=3,
            search_type="vector",
            pre_filter={"where_clause": "WHERE c.id = '1'"},
        )
        constructed_query = self.store._execute_search_query.call_args[1]["query"]
        assert "TOP @limit" in constructed_query

    # ------------------------------------------------------------------
    # Default search_type == "vector" (same as _query)
    # ------------------------------------------------------------------

    def test_default_search_type_is_vector(self) -> None:
        """_search_query without search_type behaves like old _query (vector search)."""
        self.store._search_query(vectors=[1.0, 0.0, 0.0], limit=5)
        call_kwargs = self.store._execute_search_query.call_args[1]
        assert call_kwargs["search_type"] == "vector"
        constructed_query = call_kwargs["query"]
        assert "VectorDistance" in constructed_query
        assert "ORDER BY" in constructed_query
        assert "SimilarityScore" in constructed_query

    # ------------------------------------------------------------------
    # query() public entry-point passes pre_filter through
    # ------------------------------------------------------------------

    def test_query_passes_pre_filter_through(self) -> None:
        """query() must forward pre_filter kwargs to _search_query."""
        from llama_index.core.vector_stores.types import VectorStoreQuery
        self.store._search_query = MagicMock(return_value=self._empty_result)

        vq = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3)
        self.store.query(
            vq,
            pre_filter={"where_clause": "WHERE c.author = 'King'"},
        )
        call_kwargs = self.store._search_query.call_args[1]
        assert call_kwargs.get("pre_filter") == {"where_clause": "WHERE c.author = 'King'"}

    def test_query_passes_vectors_and_limit(self) -> None:
        """query() must extract query_embedding and similarity_top_k correctly."""
        from llama_index.core.vector_stores.types import VectorStoreQuery
        self.store._search_query = MagicMock(return_value=self._empty_result)

        vq = VectorStoreQuery(query_embedding=[0.5, 0.5, 0.0], similarity_top_k=7)
        self.store.query(vq)
        call_kwargs = self.store._search_query.call_args[1]
        assert call_kwargs["vectors"] == [0.5, 0.5, 0.0]
        assert call_kwargs["limit"] == 7

    # ------------------------------------------------------------------
    # Result structure: same fields as _query returned
    # ------------------------------------------------------------------

    def test_result_has_nodes_similarities_ids(self) -> None:
        from llama_index.core.vector_stores.types import VectorStoreQueryResult
        from llama_index.core.schema import TextNode
        mock_node = TextNode(id_="abc", text="hello")
        self.store._execute_search_query = MagicMock(
            return_value=VectorStoreQueryResult(
                nodes=[mock_node], similarities=[0.9], ids=["abc"]
            )
        )
        result = self.store._search_query(
            vectors=[1.0, 0.0, 0.0], limit=1, search_type="vector"
        )
        assert result.nodes == [mock_node]
        assert result.similarities == [0.9]
        assert result.ids == ["abc"]



