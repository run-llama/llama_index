from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore


class FakeMilvusClient:
    def __init__(self, *args, **kwargs):
        pass

    def list_collections(self):
        return ["llamacollection"]


def test_class():
    names_of_base_classes = [b.__name__ for b in MilvusVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_async_client_is_created_lazily(monkeypatch):
    async_client_calls = []

    class FakeAsyncMilvusClient:
        def __init__(self, **kwargs):
            async_client_calls.append(kwargs)

    monkeypatch.setattr(
        "llama_index.vector_stores.milvus.base.MilvusClient", FakeMilvusClient
    )
    monkeypatch.setattr(
        "llama_index.vector_stores.milvus.base.AsyncMilvusClient",
        FakeAsyncMilvusClient,
    )
    monkeypatch.setattr(
        MilvusVectorStore,
        "_create_index_if_required",
        lambda self: None,
    )

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",
        token="token",
        alias="default",
        server_pem_path="/tmp/cert.pem",
    )

    assert async_client_calls == []

    assert isinstance(vector_store.aclient, FakeAsyncMilvusClient)
    assert async_client_calls == [
        {
            "uri": "http://localhost:19530",
            "token": "token",
            "server_pem_path": "/tmp/cert.pem",
        }
    ]


def test_async_client_stays_disabled(monkeypatch):
    monkeypatch.setattr(
        "llama_index.vector_stores.milvus.base.MilvusClient", FakeMilvusClient
    )
    monkeypatch.setattr(
        MilvusVectorStore,
        "_create_index_if_required",
        lambda self: None,
    )

    vector_store = MilvusVectorStore(use_async_client=False)

    assert vector_store.aclient is None
