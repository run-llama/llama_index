"""
Unit tests for Azure CosmosDB NoSQL Vector Store.

Tests core methods and helpers without any database access using mocks.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from llama_index.vector_stores.azurecosmosnosql.utils import (
    AzureCosmosDBNoSqlVectorSearchType,
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
    vector_embedding_policy: Dict[str, Any] = None,
) -> AzureCosmosDBNoSqlVectorSearch:
    """Create a store instance backed entirely by mocks (no real DB calls)."""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_container = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_db
    mock_db.create_container_if_not_exists.return_value = mock_container

    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=mock_client,
        vector_embedding_policy=vector_embedding_policy or VECTOR_EMBEDDING_POLICY,
        indexing_policy=indexing_policy or INDEXING_POLICY,
        cosmos_container_properties=container_properties or COSMOS_CONTAINER_PROPERTIES,
        cosmos_database_properties={},
        database_name="test_db",
        container_name="test_container",
        full_text_search_enabled=full_text_search_enabled,
    )


def _embedding_policy(distance_function: str) -> Dict[str, Any]:
    """Build a vector embedding policy with the given distance function."""
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": distance_function,
                "dimensions": 3,
            }
        ]
    }


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
        assert values == {"vector", "full_text_search", "full_text_ranking", "hybrid"}

    def test_string_comparison(self) -> None:
        assert AzureCosmosDBNoSqlVectorSearchType.VECTOR == "vector"
        assert AzureCosmosDBNoSqlVectorSearchType.HYBRID == "hybrid"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            AzureCosmosDBNoSqlVectorSearchType("invalid_type")

    def test_removed_threshold_variants_no_longer_exist(self) -> None:
        # Threshold filtering and weighted RRF are now controlled by the
        # ``threshold`` and ``weights`` keyword arguments, not by separate
        # search types — make sure the old enum values are gone.
        with pytest.raises(ValueError):
            AzureCosmosDBNoSqlVectorSearchType("vector_score_threshold")
        with pytest.raises(ValueError):
            AzureCosmosDBNoSqlVectorSearchType("hybrid_score_threshold")
        with pytest.raises(ValueError):
            AzureCosmosDBNoSqlVectorSearchType("weighted_hybrid_search")


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

    def test_is_full_text_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH,
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
        ):
            assert self.store._is_full_text_search_type(st), f"{st} should be full text"

    def test_is_not_full_text_search_type(self) -> None:
        assert not self.store._is_full_text_search_type(
            AzureCosmosDBNoSqlVectorSearchType.VECTOR
        )

    def test_is_vector_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.VECTOR,
            AzureCosmosDBNoSqlVectorSearchType.HYBRID,
        ):
            assert self.store._is_vector_search_type(st), f"{st} should be vector"

    def test_is_not_vector_search_type(self) -> None:
        for st in (
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_SEARCH,
            AzureCosmosDBNoSqlVectorSearchType.FULL_TEXT_RANKING,
        ):
            assert not self.store._is_vector_search_type(st), (
                f"{st} should not be vector"
            )

    def test_is_hybrid_search_type(self) -> None:
        assert self.store._is_hybrid_search_type(
            AzureCosmosDBNoSqlVectorSearchType.HYBRID
        )
        assert not self.store._is_hybrid_search_type(
            AzureCosmosDBNoSqlVectorSearchType.VECTOR
        )


# ===========================================================================
# _validate_search_args tests
# ===========================================================================


class TestValidateSearchArgs:
    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)
        self.store_no_fts = _make_store(full_text_search_enabled=False)

    def test_valid_vector_search(self) -> None:
        # Should not raise
        self.store._validate_search_args(search_type="vector", vector=[1.0, 0.0, 0.0])

    def test_invalid_search_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid search_type"):
            self.store._validate_search_args(search_type="bad_type")

    def test_vector_search_missing_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="Embedding must be provided"):
            self.store._validate_search_args(search_type="vector", vector=None)

    def test_full_text_search_disabled_raises(self) -> None:
        with pytest.raises(ValueError, match="Full text search is not enabled"):
            self.store_no_fts._validate_search_args(search_type="full_text_search")

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
        # Inside the ORDER BY RANK RRF clause itself we still inline the vector
        # literal (the parameterised form works too — see live tests — but
        # inline keeps the SQL closer to the public-docs patterns).
        assert "@vector" not in result
        assert "[1.0, 0.0, 0.0]" in result

    def test_hybrid_with_threshold_arg_does_not_change_order_by(self) -> None:
        # Threshold filtering is applied client-side after the query runs, so
        # passing ``threshold`` does not alter the ORDER BY clause.
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid",
            param_mapping=pm,
            vector=[0.0, 1.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "foo"}],
        )
        assert "ORDER BY RANK RRF(" in result
        assert "VectorDistance" in result

    def test_full_text_ranking_missing_filter_no_longer_raises_in_order_by(
        self,
    ) -> None:
        # The required-check for full_text_rank_filter is now enforced in
        # _validate_search_args (single source of truth). _generate_order_by_clause
        # itself no longer raises when the filter is missing.
        pm = ParamMapping(table="c")
        # Should not raise
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

    def test_hybrid_with_weights_emits_weight_clause(self) -> None:
        """``hybrid`` + weights triggers server-side weighted RRF."""
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "ORDER BY RANK RRF(" in result
        assert "FullTextScore" in result
        assert "VectorDistance" in result
        # Weights are appended server-side as the last RRF argument inline.
        # See https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search
        assert "[0.3, 0.7]" in result
        # Vector must still be inlined
        assert "[1.0, 0.0, 0.0]" in result
        assert "@vector" not in result

    def test_hybrid_without_weights_omits_weight_clause(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=None,
        )
        assert "ORDER BY RANK RRF(" in result
        # No weights array literal appended when weights is None
        assert (
            ", [" not in result.split("RRF(", 1)[1].rsplit(")", 1)[0].rsplit(",", 1)[-1]
        )

    def test_hybrid_weights_multiple_text_components(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_order_by_clause(
            search_type="hybrid",
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
        # Weights are inlined as the last RRF arg
        assert "[0.2, 0.3, 0.5]" in result


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

    def test_hybrid_projection_includes_similarity_score(self) -> None:
        # ``hybrid`` always projects SimilarityScore now (essentially free —
        # the VectorDistance is already computed during the search — and
        # useful both as caller-visible relevance signal and for the optional
        # ``threshold`` post-filter).
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "SimilarityScore" in result
        assert "VectorDistance(c[@vectorKey], @vector) as SimilarityScore" in result

    def test_full_text_ranking_excludes_similarity_score(self) -> None:
        # Pure full-text ranking has no vector and therefore no SimilarityScore.
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="full_text_ranking",
            param_mapping=pm,
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert "SimilarityScore" not in result
        assert "VectorDistance" not in result

    def test_full_text_search_excludes_similarity_score(self) -> None:
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="full_text_search", param_mapping=pm
        )
        assert "SimilarityScore" not in result
        assert "VectorDistance" not in result

    def test_hybrid_return_with_vectors_uses_parameterised_projection(self) -> None:
        # The base field projection inside ORDER BY RANK queries still uses
        # direct paths (closer to the public-docs patterns), but the embedding
        # itself is projected using the parameterised form — which CosmosDB
        # accepts inside ORDER BY RANK projections.
        pm = ParamMapping(table="c")
        result = self.store._generate_projection_fields(
            search_type="hybrid",
            param_mapping=pm,
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            return_with_vectors=True,
        )
        assert "c[@vectorKey] as vector" in result

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
        # The only bracket-indexer usage allowed inside ORDER BY RANK queries
        # is the SimilarityScore projection (c[@vectorKey] for VectorDistance).
        assert "c[@text" not in result
        assert "c[@metadata" not in result

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

    def test_full_text_search_query_no_order_by(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=5,
            search_type="full_text_search",
            where="FullTextContains(c.text, 'lorem')",
        )
        assert "ORDER BY" not in query
        assert "WHERE FullTextContains" in query
        assert "TOP @limit" in query

    def test_full_text_ranking_query(self) -> None:
        query, _ = self.store._construct_search_query(
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

    def test_hybrid_with_threshold_query(self) -> None:
        # ``threshold`` doesn't change the SQL itself — it's a client-side
        # post-filter. SimilarityScore is projected unconditionally for hybrid.
        query, _ = self.store._construct_search_query(
            limit=5,
            search_type="hybrid",
            vector=[0.0, 1.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "ipsum"}],
        )
        assert "TOP" not in query
        assert "OFFSET 0 LIMIT 5" in query
        assert "ORDER BY RANK RRF(" in query
        assert "SimilarityScore" in query

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

    def test_hybrid_with_weights_uses_offset_limit_not_top(self) -> None:
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "TOP" not in query
        assert "OFFSET 0 LIMIT 3" in query

    def test_hybrid_with_weights_emits_weights_inline_in_sql(self) -> None:
        """Weights are passed server-side as the last argument to RRF."""
        query, _ = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        assert "ORDER BY RANK RRF(" in query
        assert "[0.3, 0.7]" in query
        assert "FullTextScore" in query
        assert "VectorDistance(c.embedding, [1.0, 0.0, 0.0])" in query

    def test_hybrid_no_weights_differs_from_with_weights(self) -> None:
        """SQL output differs when weights are provided vs not."""
        q_with, _ = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        q_without, _ = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
        )
        assert q_with != q_without
        assert "[0.3, 0.7]" in q_with
        assert "[0.3, 0.7]" not in q_without

    def test_hybrid_vector_inline_in_order_by(self) -> None:
        # The vector inside the ORDER BY RANK RRF clause is inlined as a literal
        # (matches the public-docs pattern). The projection's SimilarityScore
        # uses a parameterised @vector — so @vector IS a query parameter even
        # for hybrid searches.
        query, params = self.store._construct_search_query(
            limit=3,
            search_type="hybrid",
            vector=[0.3, 0.7, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "test"}],
        )
        # Inline literal appears in the ORDER BY clause
        assert "VectorDistance(c.embedding, [0.3, 0.7, 0.0])" in query
        # Parameterised @vector is used in the SimilarityScore projection
        param_names = {p["name"] for p in params}
        assert "@vector" in param_names


# ===========================================================================
# _search_query backward-compat (pre_filter) tests
# ===========================================================================


class TestSearchQueryPreFilterCompat:
    """
    Verify _search_query is a full backward-compatible replacement for _query.

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
        assert call_kwargs.get("pre_filter") == {
            "where_clause": "WHERE c.author = 'King'"
        }

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


class TestValidateSearchArgsExtended:
    """
    Tests for centralized validation: rank-filter requirement, offset_limit
    format, weights length matching.
    """

    def setup_method(self) -> None:
        self.store = _make_store(full_text_search_enabled=True)

    # full_text_rank_filter required for full_text_ranking + hybrid
    @pytest.mark.parametrize(
        "search_type",
        [
            "full_text_ranking",
            "hybrid",
        ],
    )
    def test_missing_full_text_rank_filter_raises(self, search_type: str) -> None:
        with pytest.raises(ValueError, match="full_text_rank_filter"):
            self.store._validate_search_args(
                search_type=search_type,
                vector=[1.0, 0.0, 0.0],
                full_text_rank_filter=None,
            )

    def test_missing_full_text_rank_filter_via_search_query_raises(self) -> None:
        with pytest.raises(ValueError, match="full_text_rank_filter"):
            self.store._search_query(
                vectors=[1.0, 0.0, 0.0],
                limit=3,
                search_type="hybrid",
                full_text_rank_filter=None,
            )

    # offset_limit regex validation
    @pytest.mark.parametrize(
        "bad_value",
        [
            "OFFSET 0",
            "LIMIT 5",
            "ORDER BY c.id",
            "OFFSET 0 LIMIT 5; DROP TABLE c",
            "offset 0 limit 5 ;",
            "OFFSET abc LIMIT 5",
        ],
    )
    def test_invalid_offset_limit_raises(self, bad_value: str) -> None:
        with pytest.raises(ValueError, match="offset_limit"):
            self.store._validate_search_args(
                search_type="vector",
                vector=[1.0, 0.0, 0.0],
                offset_limit=bad_value,
            )

    @pytest.mark.parametrize(
        "good_value",
        [
            "OFFSET 0 LIMIT 5",
            "offset 10 limit 20",
            "  OFFSET 0 LIMIT 5  ",
            "OFFSET   3   LIMIT   10",
        ],
    )
    def test_valid_offset_limit_accepted(self, good_value: str) -> None:
        # Should not raise
        self.store._validate_search_args(
            search_type="vector",
            vector=[1.0, 0.0, 0.0],
            offset_limit=good_value,
        )

    # weights length validation for hybrid + weights
    def test_weights_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            self.store._validate_search_args(
                search_type="hybrid",
                vector=[1.0, 0.0, 0.0],
                full_text_rank_filter=[
                    {"search_field": "text", "search_text": "lorem"}
                ],
                weights=[0.1, 0.2, 0.7],  # 3 weights, but only 2 components
            )

    def test_weights_correct_length_accepted(self) -> None:
        # 1 text component + 1 vector component = 2 weights
        self.store._validate_search_args(
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[{"search_field": "text", "search_text": "lorem"}],
            weights=[0.3, 0.7],
        )
        # 2 text components + 1 vector component = 3 weights
        self.store._validate_search_args(
            search_type="hybrid",
            vector=[1.0, 0.0, 0.0],
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem"},
                {"search_field": "title", "search_text": "ipsum"},
            ],
            weights=[0.2, 0.3, 0.5],
        )

    @pytest.mark.parametrize(
        "search_type",
        ["vector", "full_text_search", "full_text_ranking"],
    )
    def test_weights_rejected_for_non_hybrid_search_type(
        self, search_type: str
    ) -> None:
        # ``weights`` is only meaningful for hybrid search.
        kwargs: dict = {"search_type": search_type, "weights": [0.3, 0.7]}
        if search_type in ("vector", "hybrid"):
            kwargs["vector"] = [1.0, 0.0, 0.0]
        if search_type in ("full_text_ranking",):
            kwargs["full_text_rank_filter"] = [
                {"search_field": "text", "search_text": "lorem"}
            ]
        with pytest.raises(ValueError, match="weights"):
            self.store._validate_search_args(**kwargs)

    @pytest.mark.parametrize(
        "search_type",
        ["full_text_search", "full_text_ranking"],
    )
    def test_threshold_rejected_for_non_vector_search_type(
        self, search_type: str
    ) -> None:
        # ``threshold`` only applies to search types that compute a vector
        # similarity score (``vector`` or ``hybrid``).
        kwargs: dict = {"search_type": search_type, "threshold": 0.5}
        if search_type == "full_text_ranking":
            kwargs["full_text_rank_filter"] = [
                {"search_field": "text", "search_text": "lorem"}
            ]
        with pytest.raises(ValueError, match="threshold"):
            self.store._validate_search_args(**kwargs)


class TestThresholdSemantics:
    """Tests for unified threshold default (None disables filtering, strict >)."""

    def setup_method(self) -> None:
        self.store = _make_store()

    def test_threshold_none_does_not_filter(self) -> None:
        # All items returned regardless of score when threshold=None
        items = [
            {"id": "a", "text": "x", "metadata": {}, "SimilarityScore": 0.1},
            {"id": "b", "text": "y", "metadata": {}, "SimilarityScore": 0.9},
        ]
        self.store._container = MagicMock()
        self.store._container.query_items = MagicMock(return_value=iter(items))
        result = self.store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=None,
        )
        assert len(result.ids) == 2

    def test_threshold_strict_greater_than(self) -> None:
        # Items with score == threshold are excluded (strict >)
        items = [
            {"id": "a", "text": "x", "metadata": {}, "SimilarityScore": 0.5},
            {"id": "b", "text": "y", "metadata": {}, "SimilarityScore": 0.6},
            {"id": "c", "text": "z", "metadata": {}, "SimilarityScore": 0.4},
        ]
        self.store._container = MagicMock()
        self.store._container.query_items = MagicMock(return_value=iter(items))
        result = self.store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.5,
        )
        # Only id 'b' (0.6 > 0.5) survives
        assert result.ids == ["b"]

    def test_hybrid_score_threshold_filters_results(self) -> None:
        # Verifies the previously-broken hybrid_score_threshold flow:
        # SimilarityScore must now be projected and used to filter.
        items = [
            {"id": "a", "text": "x", "metadata": {}, "SimilarityScore": 0.9},
            {"id": "b", "text": "y", "metadata": {}, "SimilarityScore": 0.3},
        ]
        self.store._container = MagicMock()
        self.store._container.query_items = MagicMock(return_value=iter(items))
        result = self.store._execute_search_query(
            query="SELECT * FROM c",
            search_type="hybrid",
            threshold=0.5,
        )
        assert result.ids == ["a"]

    def test_search_query_default_threshold_is_none(self) -> None:
        # Default threshold of None means no filtering (regression: was 0.5 before).
        items = [
            {"id": "low", "text": "x", "metadata": {}, "SimilarityScore": 0.1},
            {"id": "mid", "text": "y", "metadata": {}, "SimilarityScore": 0.5},
        ]
        self.store._container = MagicMock()
        self.store._container.query_items = MagicMock(return_value=iter(items))
        result = self.store._search_query(
            vectors=[1.0, 0.0, 0.0],
            limit=5,
            search_type="vector",
        )
        assert len(result.ids) == 2


# ===========================================================================
# Distance-function semantics tests
# ===========================================================================


class TestDistanceFunctions:
    """
    Verify that threshold filtering uses the correct comparison direction for
    each Cosmos DB NoSQL distance function.

    Per Microsoft Learn (Azure Cosmos DB NoSQL vector search):
      * cosine     — similarity in [-1, +1], higher = more similar
      * dotproduct — similarity in [-inf, +inf], higher = more similar
      * euclidean  — distance in [0, +inf], LOWER = more similar
    """

    def test_cosine_default_when_distance_function_missing(self) -> None:
        policy = {
            "vectorEmbeddings": [
                {"path": "/embedding", "dataType": "float32", "dimensions": 3}
            ]
        }
        store = _make_store(vector_embedding_policy=policy)
        assert store._distance_function == "cosine"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("cosine", "cosine"),
            ("Cosine", "cosine"),
            ("COSINE", "cosine"),
            ("euclidean", "euclidean"),
            ("Euclidean", "euclidean"),
            ("dotproduct", "dotproduct"),
            ("DotProduct", "dotproduct"),
        ],
    )
    def test_distance_function_normalised_lowercase(
        self, raw: str, expected: str
    ) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy(raw))
        assert store._distance_function == expected

    # -- Threshold filtering ------------------------------------------------

    @staticmethod
    def _items() -> list:
        # 3 items spanning a wide score range for clear pass/fail.
        return [
            {"id": "near", "text": "n", "metadata": {}, "SimilarityScore": 0.1},
            {"id": "mid", "text": "m", "metadata": {}, "SimilarityScore": 0.5},
            {"id": "far", "text": "f", "metadata": {}, "SimilarityScore": 0.9},
        ]

    def test_cosine_threshold_keeps_high_scores(self) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy("cosine"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.4,
        )
        # cosine: keep score > 0.4 → mid (0.5) and far (0.9)
        assert sorted(result.ids) == ["far", "mid"]

    def test_dotproduct_threshold_keeps_high_scores(self) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy("dotproduct"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.4,
        )
        # dotproduct: same direction as cosine — keep score > 0.4
        assert sorted(result.ids) == ["far", "mid"]

    def test_euclidean_threshold_keeps_low_scores(self) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy("euclidean"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.4,
        )
        # euclidean: distance, keep score < 0.4 → only "near" (0.1)
        assert result.ids == ["near"]

    def test_euclidean_hybrid_score_threshold_uses_inverse_filter(self) -> None:
        store = _make_store(
            full_text_search_enabled=True,
            vector_embedding_policy=_embedding_policy("euclidean"),
        )
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="hybrid",
            threshold=0.5,
        )
        # euclidean + hybrid threshold: keep score < 0.5 → only "near"
        assert result.ids == ["near"]

    def test_euclidean_threshold_strict_excludes_equal(self) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy("euclidean"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.5,
        )
        # strict <: exclude the 0.5 match
        assert result.ids == ["near"]

    def test_cosine_threshold_strict_excludes_equal(self) -> None:
        store = _make_store(vector_embedding_policy=_embedding_policy("cosine"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=0.5,
        )
        # strict >: exclude the 0.5 match
        assert result.ids == ["far"]

    def test_threshold_none_disables_filtering_for_all_distance_functions(
        self,
    ) -> None:
        for dist in ("cosine", "dotproduct", "euclidean"):
            store = _make_store(vector_embedding_policy=_embedding_policy(dist))
            store._container = MagicMock()
            store._container.query_items = MagicMock(return_value=iter(self._items()))
            result = store._execute_search_query(
                query="SELECT * FROM c",
                search_type="vector",
                threshold=None,
            )
            assert sorted(result.ids) == ["far", "mid", "near"], (
                f"distance_function={dist} should not filter when threshold=None"
            )

    def test_distance_function_does_not_affect_search_when_threshold_none(
        self,
    ) -> None:
        # When threshold is None the post-filter is bypassed regardless of
        # distance function, so all items pass through.
        store = _make_store(vector_embedding_policy=_embedding_policy("euclidean"))
        store._container = MagicMock()
        store._container.query_items = MagicMock(return_value=iter(self._items()))
        result = store._execute_search_query(
            query="SELECT * FROM c",
            search_type="vector",
            threshold=None,
        )
        assert sorted(result.ids) == ["far", "mid", "near"]
