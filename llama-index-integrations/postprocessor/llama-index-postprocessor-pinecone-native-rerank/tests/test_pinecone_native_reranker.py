from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.pinecone_native_rerank import PineconeNativeRerank


def test_pinecone_native_reranker():
    names_of_base_classes = [b.__name__ for b in PineconeNativeRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes
