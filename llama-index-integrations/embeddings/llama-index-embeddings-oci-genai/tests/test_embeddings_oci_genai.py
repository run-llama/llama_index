from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings


def test_oci_genai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OCIGenAIEmbeddings.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
