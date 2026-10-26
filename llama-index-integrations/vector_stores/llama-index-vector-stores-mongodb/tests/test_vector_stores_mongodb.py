from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from unittest.mock import MagicMock
import pytest

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in MongoDBAtlasVectorSearch.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture
def store() -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=MagicMock(),
        async_mongodb_client=MagicMock(),
        db_name="test_db",
        collection_name="test_collection",
    )


@pytest.mark.parametrize(
    ("alpha", "expected_alpha"),
    [
        (None, 0.5),
        (0.0, 0.0),
        (0.01, 0.01),
        (0.2, 0.2),
        (0.5, 0.5),
        (0.8, 0.8),
        (0.99, 0.99),
        (1.0, 1.0),
    ],
)
def test_hybrid_query_alpha_weights(
    store: MongoDBAtlasVectorSearch,
    alpha: float | None,
    expected_alpha: float,
) -> None:
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        query_str="test",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        alpha=alpha,
    )

    pipeline = store._create_query_pipeline(query)

    scoring_stage = next(
        stage["$addFields"]["score"]
        for stage in pipeline
        if "$addFields" in stage and "score" in stage["$addFields"]
    )

    vector_expression, text_expression = scoring_stage["$add"]
    # vector weight must preserve the supplied alpha
    assert vector_expression["$multiply"] == [
        expected_alpha,
        "$vector_score",
    ]
    # full-text weight must be 1 - alpha
    assert text_expression["$multiply"] == [
        {"$subtract": [1.0, expected_alpha]},
        "$fulltext_score",
    ]
