import pytest
from llama_index.storage.kvstore.firestore_kvstore import FirestoreKVStore

try:
    from google.cloud import firestore_v1 as firestore
except ImportError:
    firestore = None  # type: ignore


@pytest.fixture()
def kvstore_with_data(firestore_kvstore: FirestoreKVStore) -> FirestoreKVStore:
    test_key = "test_key"
    test_doc = {"test_obj_key": "test_obj_val"}
    firestore_kvstore.put(test_key, test_doc)
    return firestore_kvstore


@pytest.mark.skipif(firestore is None, reason="firestore not installed")
def test_kvstore_basic(firestore_kvstore: FirestoreKVStore) -> None:
    test_key = "test_key"
    test_doc = {"test_obj_key": "test_obj_val"}
    firestore_kvstore.put(test_key, test_doc)
    doc = firestore_kvstore.get(test_key)
    assert doc == test_doc

    doc = firestore_kvstore.get(test_key, collection="non_existent")
    assert doc is None
