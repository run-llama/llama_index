from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding


def test_dashscope_embedding_class():
    names_of_base_classes = [b.__name__ for b in DashScopeEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes
