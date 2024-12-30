from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.neptune import NeptuneAnalyticsVectorStore


def test_neptune_analytics_vector_store():
    names_of_base_classes = [b.__name__ for b in NeptuneAnalyticsVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
