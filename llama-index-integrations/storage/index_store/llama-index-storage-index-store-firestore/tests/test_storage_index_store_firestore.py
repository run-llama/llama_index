from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.index_store.firestore import FirestoreIndexStore


def test_class():
    names_of_base_classes = [b.__name__ for b in FirestoreIndexStore.__mro__]
    assert KVIndexStore.__name__ in names_of_base_classes
