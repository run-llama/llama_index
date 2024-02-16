from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore


def test_class():
    names_of_base_classes = [b.__name__ for b in MongoDocumentStore.__mro__]
    assert KVDocumentStore.__name__ in names_of_base_classes
