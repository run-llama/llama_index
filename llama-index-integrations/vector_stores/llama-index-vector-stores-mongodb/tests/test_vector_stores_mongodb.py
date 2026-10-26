from typing import Optional
from unittest.mock import MagicMock

import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in MongoDBAtlasVectorSearch.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    ("alpha", "expected_alpha"), [(0.0, 0.0), (1.0, 1.0), (None, 0.5)]
)
def test_hybrid_pipeline_alpha(alpha: Optional[float], expected_alpha: float) -> None:
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=MagicMock(),
        async_mongodb_client=MagicMock(),
        db_name="test_db",
        collection_name="test_collection",
    )
    query = VectorStoreQuery(
        mode=VectorStoreQueryMode.HYBRID,
        alpha=alpha,
        query_str="hello",
        query_embedding=[0.1, 0.2],
        similarity_top_k=2,
    )

    pipeline = vector_store._create_query_pipeline(query)

    score_stage = next(
        stage["$addFields"]["score"]["$add"]
        for stage in pipeline
        if "score" in stage.get("$addFields", {})
    )
    assert score_stage == [
        {"$multiply": [expected_alpha, "$vector_score"]},
        {"$multiply": [{"$subtract": [1.0, expected_alpha]}, "$fulltext_score"]},
    ]
