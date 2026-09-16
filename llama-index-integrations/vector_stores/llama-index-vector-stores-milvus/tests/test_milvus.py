import sys
import types

if "pymilvus" not in sys.modules:
    pymilvus = types.ModuleType("pymilvus")

    class Function:
        def __init__(self, *args, **kwargs):
            pass

    class FunctionType:
        pass

    class CollectionSchema:
        pass

    class MilvusClient:
        pass

    class AsyncMilvusClient:
        pass

    class DataType:
        pass

    class AnnSearchRequest:
        pass

    class WeightedRanker:
        pass

    class RRFRanker:
        pass

    pymilvus.Function = Function
    pymilvus.FunctionType = FunctionType
    pymilvus.CollectionSchema = CollectionSchema
    pymilvus.MilvusClient = MilvusClient
    pymilvus.AsyncMilvusClient = AsyncMilvusClient
    pymilvus.DataType = DataType
    pymilvus.AnnSearchRequest = AnnSearchRequest
    pymilvus.WeightedRanker = WeightedRanker
    pymilvus.RRFRanker = RRFRanker
    sys.modules["pymilvus"] = pymilvus

    types_mod = types.ModuleType("pymilvus.client.types")

    class LoadState:
        pass

    types_mod.LoadState = LoadState
    sys.modules["pymilvus.client.types"] = types_mod

    index_mod = types.ModuleType("pymilvus.milvus_client.index")

    class IndexParams:
        pass

    index_mod.IndexParams = IndexParams
    sys.modules["pymilvus.milvus_client.index"] = index_mod

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import _to_milvus_filter
from llama_index.vector_stores.milvus.utils import parse_standard_filters


def test_parse_standard_filters_empty_and_none():
    # standard_filters is None
    filters, expr = parse_standard_filters(None)
    assert filters == []
    assert expr == ""

    # standard_filters with empty list
    filters, expr = parse_standard_filters(MetadataFilters(filters=[]))
    assert filters == []
    assert expr == ""

    # standard_filters with condition=None
    filters_obj = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="news", operator=FilterOperator.EQ
            )
        ]
    )
    filters_obj.condition = None
    filters, expr = parse_standard_filters(filters_obj)
    assert filters == ["category == 'news'"]
    assert expr == "category == 'news'"


def test_parse_standard_filters_nested_empty():
    # Nested empty filters
    nested = MetadataFilters(filters=[MetadataFilters(filters=[])])
    filters, expr = parse_standard_filters(nested)
    assert filters == []
    assert expr == ""


def test_to_milvus_filter_empty():
    assert _to_milvus_filter(MetadataFilters(filters=[])) == ""
    assert _to_milvus_filter(None) == ""


def test_prepare_before_search_empty_filters_with_node_ids():
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    object.__setattr__(store, "doc_id_field", "doc_id")
    object.__setattr__(store, "output_fields", [])
    object.__setattr__(store, "text_key", "text")

    query = VectorStoreQuery(
        filters=MetadataFilters(filters=[]),
        node_ids=["node-123"],
    )

    string_expr, output_fields = store._prepare_before_search(query)
    assert not string_expr.startswith(" and ")
    assert string_expr == 'id in ["node-123"]'


def test_prepare_before_search_with_filters_and_node_ids():
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    object.__setattr__(store, "doc_id_field", "doc_id")
    object.__setattr__(store, "output_fields", [])
    object.__setattr__(store, "text_key", "text")

    query = VectorStoreQuery(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="books", operator=FilterOperator.EQ
                )
            ]
        ),
        node_ids=["node-123"],
    )

    string_expr, output_fields = store._prepare_before_search(query)
    assert string_expr == "category == 'books' and id in [\"node-123\"]"
