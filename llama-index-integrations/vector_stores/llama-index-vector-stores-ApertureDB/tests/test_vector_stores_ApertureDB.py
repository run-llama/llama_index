from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.ApertureDB import ApertureDBVectorStore

import aperturedb


class MockConnector:
    def __init__(self, exists) -> None:
        self.exists = exists
        self.queries = []

    def clone(self):
        return self

    def query(self, *args, **kwargs):
        print("query called with args:", args, "and kwargs:", kwargs)
        self.queries.append(args[0])
        response, blobs = [], []
        if "FindDescriptorSet" in args[0][0]:
            response = [
                {
                    "FindDescriptorSet": {
                        "response": 0,
                        "status": 0,
                        "result": 0,
                    }
                }
            ]
            if self.exists:
                response[0]["FindDescriptorSet"]["entities"] = [
                    {
                        "_engines": ["HNSW"],
                        "_metrics": ["CS"],
                        "_dimensions": 1024,
                    }
                ]
            else:
                response[0]["FindDescriptorSet"]["entities"] = []
        print("query response:", response)
        return response, blobs

    def last_query_ok(self):
        return True

    def get_last_response_str(self):
        return "response"


def test_class():
    names_of_base_classes = [b.__name__ for b in ApertureDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_vector_store_initialize_noop_when_DS_exists(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(True),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    assert store._dimensions == 1024
    expected_queries = ["GetStatus", "FindDescriptorSet"]
    print(store.client.queries)
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]


def test_vector_store_initialize_noop_when_DS_does_not_exist(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(False),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    assert store._dimensions == 1024
    expected_queries = [
        "GetStatus",
        "FindDescriptorSet",
        "AddDescriptorSet",
        "CreateIndex",
        "CreateIndex",
        "CreateIndex",
    ]
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]
