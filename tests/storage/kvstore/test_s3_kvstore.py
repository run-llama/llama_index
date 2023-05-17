from typing import Generator
import pytest
from llama_index.storage.kvstore.s3_kvstore import S3DBKVStore

try:
    import boto3
    from moto import mock_s3

    has_boto_libs = True
except ImportError:
    has_boto_libs = False


@pytest.fixture()
def kvstore_from_mocked_bucket() -> Generator[S3DBKVStore, None, None]:
    with mock_s3():
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("test_bucket")
        bucket.create()
        yield S3DBKVStore(bucket)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_put_get(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore_from_mocked_bucket.put(test_key, test_blob)
    blob = kvstore_from_mocked_bucket.get(test_key)
    assert blob == test_blob


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_get_non_existent(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key = "test_key"
    blob = kvstore_from_mocked_bucket.get(test_key)
    assert blob is None


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_put_get_multiple_collections(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key = "test_key"
    test_blob_collection_a = {"test_obj_key": "a"}
    test_blob_collection_b = {"test_obj_key": "b"}
    kvstore_from_mocked_bucket.put(
        test_key, test_blob_collection_a, collection="test_collection_a"
    )
    kvstore_from_mocked_bucket.put(
        test_key, test_blob_collection_b, collection="test_collection_b"
    )
    blob_collection_a = kvstore_from_mocked_bucket.get(
        test_key, collection="test_collection_a"
    )
    blob_collection_b = kvstore_from_mocked_bucket.get(
        test_key, collection="test_collection_b"
    )
    assert test_blob_collection_a == blob_collection_a
    assert test_blob_collection_b == blob_collection_b


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_delete(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore_from_mocked_bucket.put(test_key, test_blob)
    blob = kvstore_from_mocked_bucket.get(test_key)
    assert blob == test_blob
    assert kvstore_from_mocked_bucket.delete(test_key)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_delete_non_existent(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore_from_mocked_bucket.put(test_key, test_blob)
    assert kvstore_from_mocked_bucket.delete("wrong_key") is False


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_get_all(kvstore_from_mocked_bucket: S3DBKVStore) -> None:
    test_key_a = "test_key_a"
    test_blob_a = {"test_obj_key": "test_obj_val_a"}

    test_key_b = "test_key_b"
    test_blob_b = {"test_obj_key": "test_obj_val_b"}
    kvstore_from_mocked_bucket.put(test_key_a, test_blob_a)
    kvstore_from_mocked_bucket.put(test_key_b, test_blob_b)
    blobs = kvstore_from_mocked_bucket.get_all()

    assert blobs == {test_key_a: test_blob_a, test_key_b: test_blob_b}
