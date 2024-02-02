from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.clarifai import ClarifaiEmbedding


def test_anyscale_class():
    emb = ClarifaiEmbedding(
        pat="fake-pat",
        model_name="dummy-model-name",
        app_id="fake-app-id",
        user_id="fake-user-id",
    )
    assert isinstance(emb, BaseEmbedding)
