import pytest
import subprocess
import os
from typing import Generator
from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.storage.index_store.gel import (
    GelIndexStore,
)
from llama_index.storage.kvstore.gel import GelKVStore

try:
    import gel  # noqa

    no_packages = False
except ImportError:
    no_packages = True

skip_in_cicd = os.environ.get("CI") is not None
try:
    if not skip_in_cicd:
        subprocess.run(["gel", "project", "init", "--non-interactive"], check=True)
except subprocess.CalledProcessError as e:
    print(e)


@pytest.fixture()
def gel_kvstore() -> Generator[GelKVStore, None, None]:
    kvstore = None
    try:
        kvstore = GelKVStore()
        yield kvstore
    finally:
        if kvstore:
            keys = kvstore.get_all().keys()
            for key in keys:
                kvstore.delete(key)


@pytest.fixture()
def gel_indexstore(gel_kvstore: GelKVStore) -> GelIndexStore:
    return GelIndexStore(gel_kvstore=gel_kvstore)


@pytest.mark.skipif(no_packages or skip_in_cicd, reason="gel not installed")
def test_gel_index_store(gel_indexstore: GelIndexStore) -> None:
    index_struct = IndexGraph()
    index_store = gel_indexstore

    index_store.add_index_struct(index_struct)
    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
