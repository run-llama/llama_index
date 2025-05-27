from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.vertex_endpoint import VertexEndpointEmbedding


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in VertexEndpointEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
