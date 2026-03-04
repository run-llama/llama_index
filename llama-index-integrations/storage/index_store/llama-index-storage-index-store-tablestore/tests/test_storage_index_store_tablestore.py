import os

import pytest
from llama_index.core.data_structs import IndexGraph
from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore

from llama_index.storage.index_store.tablestore import TablestoreIndexStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TablestoreIndexStore.__mro__]
    assert KVIndexStore.__name__ in names_of_base_classes
    assert TablestoreIndexStore.__name__ in names_of_base_classes


# noinspection DuplicatedCode
@pytest.fixture()
def tablestore_index_store() -> TablestoreIndexStore:
    end_point = os.getenv("tablestore_end_point")
    instance_name = os.getenv("tablestore_instance_name")
    access_key_id = os.getenv("tablestore_access_key_id")
    access_key_secret = os.getenv("tablestore_access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )

    # 1. create tablestore vector store
    index = TablestoreIndexStore.from_config(
        endpoint=os.getenv("tablestore_end_point"),
        instance_name=os.getenv("tablestore_instance_name"),
        access_key_id=os.getenv("tablestore_access_key_id"),
        access_key_secret=os.getenv("tablestore_access_key_secret"),
    )
    index.delete_all_index()
    return index


def test_postgres_index_store(tablestore_index_store: TablestoreIndexStore) -> None:
    index_struct = IndexGraph()
    index_store = tablestore_index_store

    index_store.add_index_struct(index_struct)
    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
    all_structs = index_store.index_structs()
    assert len(all_structs) >= 1
