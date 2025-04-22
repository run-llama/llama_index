import subprocess
from typing import Generator

import pytest
from llama_index.storage.kvstore.gel import GelKVStore

try:
    import gel  # noqa

    no_packages = False
except ImportError:
    no_packages = True

try:
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


@pytest.mark.skipif(no_packages, reason="gel package not installed")
def test_kvstore_basic(gel_kvstore: GelKVStore) -> None:
    test_key = "test_key_basic"
    test_blob = {"test_obj_key": "test_obj_val"}
    gel_kvstore.put(test_key, test_blob)
    blob = gel_kvstore.get(test_key)
    assert blob == test_blob

    blob = gel_kvstore.get(test_key, collection="non_existent")
    assert blob is None

    deleted = gel_kvstore.delete(test_key)
    assert deleted


@pytest.mark.skipif(no_packages, reason="gel package not installed")
@pytest.mark.asyncio()
async def test_kvstore_async_basic(gel_kvstore: GelKVStore) -> None:
    test_key = "test_key_basic"
    test_blob = {"test_obj_key": "test_obj_val"}
    await gel_kvstore.aput(test_key, test_blob)
    blob = await gel_kvstore.aget(test_key)
    assert blob == test_blob

    blob = await gel_kvstore.aget(test_key, collection="non_existent")
    assert blob is None

    deleted = await gel_kvstore.adelete(test_key)
    assert deleted


@pytest.mark.skipif(no_packages, reason="gel package not installed")
def test_kvstore_delete(gel_kvstore: GelKVStore) -> None:
    test_key = "test_key_delete"
    test_blob = {"test_obj_key": "test_obj_val"}
    gel_kvstore.put(test_key, test_blob)
    blob = gel_kvstore.get(test_key)
    assert blob == test_blob

    gel_kvstore.delete(test_key)
    blob = gel_kvstore.get(test_key)
    assert blob is None


@pytest.mark.skipif(no_packages, reason="gel package not installed")
@pytest.mark.asyncio()
async def test_kvstore_adelete(gel_kvstore: GelKVStore) -> None:
    test_key = "test_key_delete"
    test_blob = {"test_obj_key": "test_obj_val"}
    await gel_kvstore.aput(test_key, test_blob)
    blob = await gel_kvstore.aget(test_key)
    assert blob == test_blob

    await gel_kvstore.adelete(test_key)
    blob = await gel_kvstore.aget(test_key)
    assert blob is None


@pytest.mark.skipif(no_packages, reason="gel package not installed")
def test_kvstore_getall(gel_kvstore: GelKVStore) -> None:
    test_key_1 = "test_key_1"
    test_blob_1 = {"test_obj_key": "test_obj_val"}
    gel_kvstore.put(test_key_1, test_blob_1)
    blob = gel_kvstore.get(test_key_1)
    assert blob == test_blob_1
    test_key_2 = "test_key_2"
    test_blob_2 = {"test_obj_key": "test_obj_val"}
    gel_kvstore.put(test_key_2, test_blob_2)
    blob = gel_kvstore.get(test_key_2)
    assert blob == test_blob_2

    blob = gel_kvstore.get_all()
    assert len(blob) == 2

    gel_kvstore.delete(test_key_1)
    gel_kvstore.delete(test_key_2)


@pytest.mark.skipif(no_packages, reason="gel package not installed")
@pytest.mark.asyncio()
async def test_kvstore_agetall(gel_kvstore: GelKVStore) -> None:
    test_key_1 = "test_key_1"
    test_blob_1 = {"test_obj_key": "test_obj_val"}
    await gel_kvstore.aput(test_key_1, test_blob_1)
    blob = await gel_kvstore.aget(test_key_1)
    assert blob == test_blob_1
    test_key_2 = "test_key_2"
    test_blob_2 = {"test_obj_key": "test_obj_val"}
    await gel_kvstore.aput(test_key_2, test_blob_2)
    blob = await gel_kvstore.aget(test_key_2)
    assert blob == test_blob_2

    blob = await gel_kvstore.aget_all()
    assert len(blob) == 2

    await gel_kvstore.adelete(test_key_1)
    await gel_kvstore.adelete(test_key_2)


@pytest.mark.skipif(no_packages, reason="gel package not installed")
@pytest.mark.asyncio()
async def test_kvstore_putall(gel_kvstore: GelKVStore) -> None:
    test_key = "test_key_putall_1"
    test_blob = {"test_obj_key": "test_obj_val"}
    test_key2 = "test_key_putall_2"
    test_blob2 = {"test_obj_key2": "test_obj_val2"}
    await gel_kvstore.aput_all([(test_key, test_blob), (test_key2, test_blob2)])
    blob = await gel_kvstore.aget(test_key)
    assert blob == test_blob
    blob = await gel_kvstore.aget(test_key2)
    assert blob == test_blob2

    await gel_kvstore.adelete(test_key)
    await gel_kvstore.adelete(test_key2)
