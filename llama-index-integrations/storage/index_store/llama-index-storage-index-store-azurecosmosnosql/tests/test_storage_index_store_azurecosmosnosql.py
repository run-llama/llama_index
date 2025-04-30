from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.index_store.azurecosmosnosql import AzureCosmosNoSqlIndexStore


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureCosmosNoSqlIndexStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes
