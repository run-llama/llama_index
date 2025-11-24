from unittest.mock import MagicMock
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)


def test_query_default_mode_with_filter() -> None:
    # Mock the MongoDB client and collection
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    # Setup the mock collection to return some results
    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = [
        {
            "_id": "123",
            "embedding": [0.1, 0.2],
            "text": "test text",
            "metadata": {"year": 2021},
            "score": 0.9,
        }
    ]
    mock_collection.aggregate.return_value = mock_cursor

    # Initialize the vector store
    store = MongoDBAtlasVectorSearch(
        mongodb_client=mock_client, db_name="test_db", collection_name="test_collection"
    )

    # Create a query with filters
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        mode=VectorStoreQueryMode.DEFAULT,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.GT)]
        ),
    )

    # Execute the query
    result = store.query(query)

    # Verify that aggregate was called
    assert mock_collection.aggregate.called

    # Verify the pipeline passed to aggregate
    call_args = mock_collection.aggregate.call_args
    pipeline = call_args[0][0]

    # Check if $vectorSearch stage is present and has the filter
    vector_search_stage = pipeline[0]["$vectorSearch"]
    assert "filter" in vector_search_stage
    assert vector_search_stage["filter"] == {"metadata.year": {"$gt": 2020}}


def test_query_text_search_mode() -> None:
    # Mock the MongoDB client and collection
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = []
    mock_collection.aggregate.return_value = mock_cursor

    store = MongoDBAtlasVectorSearch(mongodb_client=mock_client)

    query = VectorStoreQuery(
        query_str="test query",
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.EQ)]
        ),
    )

    store.query(query)

    assert mock_collection.aggregate.called
    pipeline = mock_collection.aggregate.call_args[0][0]

    # Check if $search stage is present
    search_stage = pipeline[0]["$search"]
    assert "compound" in search_stage
    # Check if filter is applied
    assert any(
        clause.get("equals") and clause["equals"]["value"] == 2020
        for clause in search_stage["compound"]["must"]
    )
