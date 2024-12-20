from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.alibabacloud_aisearch import AlibabaCloudAISearchEmbedding


def test_class():
    names_of_base_classes = [b.__name__ for b in AlibabaCloudAISearchEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
