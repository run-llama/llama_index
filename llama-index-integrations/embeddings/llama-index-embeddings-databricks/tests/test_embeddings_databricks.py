from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.databricks import DatabricksEmbedding


def test_dashscope_embedding_class():
    names_of_base_classes = [b.__name__ for b in DatabricksEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
