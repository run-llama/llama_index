from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gigachat import GigaChatEmbedding


def test_initialize_failure_auth_data():
    try:
        GigaChatEmbedding(auth_data=None, scope="GIGACHAT_API_PERS")
    except ValueError as e:
        assert (
            str(e)
            == "You must provide an AUTH DATA to use GigaChat. You can either pass it in as an argument or set it `GIGACHAT_AUTH_DATA`."
        )


def test_initialize_failure_scope_none():
    try:
        GigaChatEmbedding(auth_data="dummy_auth_data", scope=None)
    except ValueError as e:
        assert (
            str(e)
            == "GigaChat scope cannot be 'None'. Set 'GIGACHAT_API_PERS' for personal use or 'GIGACHAT_API_CORP' for corporate use."
        )


def test_embedding_function():
    names_of_base_classes = [b.__name__ for b in GigaChatEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
