"""
Demonstrates how to use filters in Vector Search.

If one wishes to use filter by fields in vectorSearch,
these fields also need to be indexed. These are full-text 'Atlas Search' Indexes.
Indexes can be done manually on all clusters through the Atlas UI.
For dedicated clusters (>= M10), utility functions can be used as well.
e.g. `llama_index.vector_stores.mongodb.index.create_vector_search_index`

Note that search index commands are only supported on Atlas Clusters >=M10.
"""

import os
from typing import List

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch, index, pipelines
from pymongo import MongoClient
from pymongo.collection import Collection

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
COLLECTION_NAME = "llama_index_test_filters"
VECTOR_INDEX_NAME = "vector_index"
FILTER_INDEX_NAME = "metadata_year"  # note. cannot include '.' in name

TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture()
def text_nodes() -> List[TextNode]:
    return [
        TextNode(
            text="Winter in the mountains.",
            metadata={"year": 2024, "country": "France"},
            embedding=np.ones(DIMENSIONS).tolist(),
        ),
        TextNode(
            text="Winter in the mountains.",
            metadata={"year": 2023, "country": "Canada"},
            embedding=np.ones(DIMENSIONS).tolist(),
        ),
        TextNode(
            text="Winter in the mountains.",
            metadata={"year": 2022, "country": "Chile"},
            embedding=np.ones(DIMENSIONS).tolist(),
        ),
    ]


@pytest.fixture()
def vector_store(atlas_client: MongoClient) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        vector_index_name=VECTOR_INDEX_NAME,
        fulltext_index_name=FILTER_INDEX_NAME,
    )


@pytest.fixture()
def collection(
    vector_store: MongoDBAtlasVectorSearch, text_nodes: List[TextNode]
) -> Collection:
    """Depending on uri, this could point to any type of cluster."""
    clxn = vector_store.collection
    if clxn.count_documents({}) != len(text_nodes):
        clxn.delete_many({})
        vector_store.add(text_nodes)
    assert clxn.count_documents({}) == len(text_nodes)
    return clxn


@pytest.fixture()
def vector_indexed(collection: Collection) -> str:
    """
    This creates a vector search index
    with a filter on the year field within the metadata document.

    To be able to filter on another field, said field must be indexed.
    That is done in `year_indexed` fixture. It is not a vector search type.

    Note naming: `metadata.naming`, MQL for accessing nested documents.
    """
    if not any(
        idx["name"] == VECTOR_INDEX_NAME for idx in collection.list_search_indexes()
    ):
        index.create_vector_search_index(
            collection=collection,
            index_name=VECTOR_INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
            filters=["metadata.year"],
            wait_until_complete=TIMEOUT,
        )
    return VECTOR_INDEX_NAME


@pytest.fixture()
def year_indexed(collection: Collection) -> str:
    """
    Search Index on metadata.year nested field of type number.

    This is required to do filtered vector search.
    """
    if not any(
        idx["name"] == "metadata_year" for idx in collection.list_search_indexes()
    ):
        index.create_fulltext_search_index(
            collection=collection,
            index_name="metadata_year",
            field="metadata.year",
            field_type="number",
            wait_until_complete=TIMEOUT,
        )
        assert any(
            idx["name"] == "metadata_year" for idx in collection.list_search_indexes()
        )

    return FILTER_INDEX_NAME


@pytest.fixture()
def metadata_filters() -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(key="metadata.year", operator="<=", value=2024),
            MetadataFilter(key="metadata.year", operator=FilterOperator.GT, value=2022),
        ]
    )


def test_metadata_filters(metadata_filters: MetadataFilters) -> None:
    """Test conversion of MetadataFilters to MQL."""
    mql_filters = pipelines.filters_to_mql(metadata_filters)
    assert mql_filters == {
        "$and": [{"metadata.year": {"$lte": 2024}}, {"metadata.year": {"$gt": 2022}}]
    }


@pytest.mark.skipif(
    os.environ.get("MONGODB_URI") is None,
    reason="Requires MONGODB_URI in os.environ",
    allow_module_level=True,
)
def test_search_with_filter(
    collection: Collection,
    vector_store: MongoDBAtlasVectorSearch,
    vector_indexed: str,
    year_indexed: str,
    metadata_filters: MetadataFilters,
) -> None:
    """
    Tests vector search with a filter.

    similarity_top_k=3 > len(vector_store.query(query).nodes)
    """
    query = VectorStoreQuery(
        query_embedding=(np.ones(DIMENSIONS) * 0.5).tolist(),
        query_str="winter",
        filters=metadata_filters,
        mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=3,
    )
    result = vector_store.query(query)
    assert len(result.nodes) == 2
    result_metadata = [result.nodes[i].metadata for i in range(len(result.nodes))]
    assert {"year": 2024, "country": "France"} in result_metadata
    assert {"year": 2023, "country": "Canada"} in result_metadata
    assert {"year": 2022, "country": "Chile"} not in result_metadata
