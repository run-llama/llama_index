from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding


def test_embedding_class():
    emb = VoyageEmbedding(model_name="", voyage_api_key="NOT_A_VALID_KEY")
    assert isinstance(emb, BaseEmbedding)


def test_embedding_class_with_default_model():
    emb = VoyageEmbedding(voyage_api_key="NOT_A_VALID_KEY")
    assert isinstance(emb, BaseEmbedding)


def test_voyageai_embedding_class():
    names_of_base_classes = [b.__name__ for b in VoyageEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
