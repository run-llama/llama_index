from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.vllm import VllmEmbedding

def test_vllmembedding_class():
    names_of_base_classes = [b.__name__ for b in Vllm.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_embedding_retry():
    embed_model = VllmEmbedding(
        model_name="facebook/opt-350m",
    )

    # Test successful embedding
    result = embed_model._embed(["This is a test sentence"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])
