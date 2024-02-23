import os

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding


def test_embedding_class():
    emb = VoyageEmbedding(model_name="", voyage_api_key="NOT_AVALID_KEY")
    assert isinstance(emb, BaseEmbedding)


def test_embedding_class_with_default_model():
    emb = VoyageEmbedding(voyage_api_key="NOT_AVALID_KEY")
    assert isinstance(emb, BaseEmbedding)


def test_embedding_class_with_default_model_env_api_key():
    os.environ["VOYAGE_API_KEY"] = "NOT_AVALID_KEY"
    emb = VoyageEmbedding()
    assert isinstance(emb, BaseEmbedding)
