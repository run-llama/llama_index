"""
Test of utility functions for working with Search Index commands.

Note that search index commands are only supported on Atlas Clusters >=M10.
"""

import os
from typing import Generator, List, Optional

import pytest
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch, index
from pymongo import MongoClient
from pymongo.collection import Collection

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
COLLECTION_NAME = "test_index_commands"
VECTOR_INDEX_NAME = "vector_index"
FULLTEXT_INDEX_NAME = "fulltext_index"
FILTER_FIELD_NAME = "country"
FILTER_FIELD_TYPE = "string"
TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture()
def collection(vector_store) -> Generator:
    """Depending on uri, this could point to any type of cluster."""
    clxn = vector_store.collection
    clxn.insert_many([{"year": 2024}, {"country": "Canada"}])
    yield clxn
    clxn.drop()


@pytest.fixture()
def vector_store(atlas_client: MongoClient) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        vector_index_name=VECTOR_INDEX_NAME,
    )


@pytest.mark.skipif(
    os.environ.get("MONGODB_URI") is None, reason="Requires MONGODB_URI in os.environ"
)
def test_search_index_commands_standalone(collection: Collection) -> None:
    """Tests create, update, and drop index utility functions."""
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: Optional[List[str]] = None
    wait_until_complete = TIMEOUT

    for index_info in collection.list_search_indexes():
        index.drop_vector_search_index(
            collection, index_info["name"], wait_until_complete=wait_until_complete
        )
    assert len(list(collection.list_search_indexes())) == 0

    # Create a Vector Search Index on index_name
    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    # Update that index by adding a filter
    # This will additionally index the "bar" and "foo"  fields
    # The Update method is not yet supported in Atlas Local.
    if "mongodb+srv" in os.environ.get("MONGODB_URI"):
        new_similarity = "euclidean"
        index.update_vector_search_index(
            collection=collection,
            index_name=index_name,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity=new_similarity,
            filters=[FILTER_FIELD_NAME],
            wait_until_complete=wait_until_complete,
        )

        indexes = list(collection.list_search_indexes())
        assert len(indexes) == 1
        assert indexes[0]["name"] == index_name
        fields = indexes[0]["latestDefinition"]["fields"]
        assert len(fields) == 2
        assert {"type": "filter", "path": FILTER_FIELD_NAME} in fields
        assert {
            "numDimensions": DIMENSIONS,
            "path": "embedding",
            "similarity": f"{new_similarity}",
            "type": "vector",
        } in fields

    # Now add a full-text search index for the filter field
    index.create_fulltext_search_index(
        collection=collection,
        index_name=FULLTEXT_INDEX_NAME,
        field=FILTER_FIELD_NAME,
        field_type=FILTER_FIELD_TYPE,
        wait_until_complete=TIMEOUT,
    )

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 2
    assert any(idx["name"] == FULLTEXT_INDEX_NAME for idx in indexes)
    idx_fulltext = (
        indexes[0] if indexes[0]["name"] == FULLTEXT_INDEX_NAME else indexes[1]
    )
    assert idx_fulltext["type"] == "search"
    fields = idx_fulltext["latestDefinition"]["mappings"]["fields"]
    assert fields[FILTER_FIELD_NAME]["type"] == FILTER_FIELD_TYPE

    # Finally, drop the index
    for name in [FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME]:
        index.drop_vector_search_index(
            collection, name, wait_until_complete=wait_until_complete
        )

    indexes = list(collection.list_search_indexes())
    for idx in indexes:
        assert idx["status"] == "DELETING"
