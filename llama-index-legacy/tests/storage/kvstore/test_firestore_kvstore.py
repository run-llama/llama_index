import pytest
from llama_index.legacy.storage.kvstore.firestore_kvstore import FirestoreKVStore

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


@pytest.mark.asyncio()
@pytest.mark.skipif(firestore is None, reason="firestore not installed")
async def test_kvstore_async(firestore_kvstore: FirestoreKVStore) -> None:
    test_key = "test_key"
    test_doc = {"test_obj_key": "test_obj_val"}
    await firestore_kvstore.aput(test_key, test_doc)
    doc = await firestore_kvstore.aget(test_key)
    assert doc == test_doc

    doc = await firestore_kvstore.aget(test_key, collection="non_existent")
    assert doc is None


@pytest.mark.skipif(firestore is None, reason="firestore not installed")
def test_kvstore_putall(firestore_kvstore: FirestoreKVStore) -> None:
    batch = [
        ("batch_test_key_1", {"test_obj_key_1": "test_obj_val_1"}),
        ("batch_test_key_2", {"test_obj_key_2": "test_obj_val_2"}),
    ]
    firestore_kvstore.put_all(batch)
    assert firestore_kvstore.get("batch_test_key_1") == batch[0][1]
    assert firestore_kvstore.get("batch_test_key_2") == batch[1][1]
