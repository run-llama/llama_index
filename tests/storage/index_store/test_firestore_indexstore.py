from typing import List

import pytest

from llama_index.storage.index_store.firestore_indexstore import FirestoreIndexStore
from llama_index.data_structs.data_structs import IndexGraph
from llama_index.storage.kvstore.firestore_kvstore import FirestoreKVStore

try:
    from google.cloud import firestore_v1 as firestore
except ImportError:
    firestore = None  # type: ignore


@pytest.fixture()
def firestore_indexstore(firestore_kvstore: FirestoreKVStore) -> FirestoreIndexStore:
    return FirestoreIndexStore(firestore_kvstore=firestore_kvstore)


@pytest.mark.skipif(firestore is None, reason="firestore not installed")
def test_firestore_docstore(firestore_indexstore: FirestoreIndexStore) -> None:
    index_struct = IndexGraph()
    index_store = firestore_indexstore

    index_store.add_index_struct(index_struct)
    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
