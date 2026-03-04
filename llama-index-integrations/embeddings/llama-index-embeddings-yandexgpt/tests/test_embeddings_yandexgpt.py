from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding


def test_embedding_function():
    names_of_base_classes = [b.__name__ for b in YandexGPTEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_init():
    emb = YandexGPTEmbedding(api_key="test-api", folder_id="test-folder")
    assert emb.sleep_interval == 0.1
    emb = YandexGPTEmbedding(
        api_key="test-api", folder_id="test-folder", sleep_interval=10.0
    )
    assert emb.sleep_interval == 10.0
