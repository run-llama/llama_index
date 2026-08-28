import sys
import types

import numpy as np

vearch_cluster = types.ModuleType("vearch_cluster")
vearch_cluster.VearchCluster = object
sys.modules.setdefault("vearch_cluster", vearch_cluster)

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.vearch import VearchVectorStore
import llama_index.vector_stores.vearch.base as vearch_base


class FakeVearchClient:
    def __init__(self) -> None:
        self.last_search = None
        self.delete_calls = []
        self.get_space_calls = []

    def get_space(self, db_name, table_name):
        self.get_space_calls.append((db_name, table_name))
        return ["ref_doc_id", "text", "text_embedding", "topic"]

    def search(self, db_name, table_name, query_data):
        self.last_search = (db_name, table_name, query_data)
        return {"hits": {"hits": []}}

    def delete_by_query(self, db_name, table_name, queries):
        self.delete_calls.append((db_name, table_name, queries))


def build_store(monkeypatch, fake_client, **kwargs):
    monkeypatch.setattr(
        vearch_base.vearch_cluster, "VearchCluster", lambda path_or_url: fake_client
    )
    return VearchVectorStore(
        path_or_url="http://localhost:9001",
        table_name="test_table",
        db_name="test_db",
        **kwargs,
    )


def test_class():
    names_of_base_classes = [b.__name__ for b in VearchVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_constructor_persists_flag_and_cluster_query_uses_space_fields(monkeypatch):
    fake_client = FakeVearchClient()
    store = build_store(monkeypatch, fake_client, flag=1)

    store.query(
        VectorStoreQuery(query_embedding=np.array([1.0, 0.0]), similarity_top_k=3)
    )

    assert store.flag == 1
    assert fake_client.get_space_calls == [("test_db", "test_table")]
    assert fake_client.last_search[2]["fields"] == ["ref_doc_id", "text", "topic"]


def test_standalone_query_uses_initialized_fields_without_get_space(monkeypatch):
    fake_client = FakeVearchClient()
    store = build_store(monkeypatch, fake_client, flag=0)
    store._get_matadata_field([{"topic": "vearch"}])

    store.query(
        VectorStoreQuery(query_embedding=np.array([1.0, 0.0]), similarity_top_k=2)
    )

    assert fake_client.get_space_calls == []
    assert fake_client.last_search[2]["fields"] == ["ref_doc_id", "text", "topic"]


def test_delete_uses_bound_client_delete_by_query(monkeypatch):
    fake_client = FakeVearchClient()
    store = build_store(monkeypatch, fake_client)

    store.delete("doc-1")

    assert fake_client.delete_calls == [
        (
            "test_db",
            "test_table",
            {
                "query": {
                    "filter": [{"term": {"ref_doc_id": ["doc-1"], "operator": "and"}}]
                },
                "size": 10000,
            },
        )
    ]
