import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding


def test_openai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenAILikeEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_init():
    embed_model = OpenAILikeEmbedding(
        model_name="model-name",
        api_key="fake",
        api_base="http://localhost:1234/v1",
        embed_batch_size=1,
    )
    assert embed_model.model_name == "model-name"
    assert embed_model.api_key == "fake"
    assert embed_model.api_base == "http://localhost:1234/v1"
    assert embed_model.embed_batch_size == 1

    with pytest.raises(TypeError):
        embed_model = OpenAILikeEmbedding(
            model="model-name",
        )
