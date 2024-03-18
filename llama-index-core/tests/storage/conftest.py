import pytest
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore


@pytest.fixture()
def simple_kvstore() -> SimpleKVStore:
    return SimpleKVStore()
