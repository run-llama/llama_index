from typing import Generator
import pytest
from pytest import MonkeyPatch
from llama_index.storage.kvstore.dynamodb_kvstore import DynamoDBKVStore

try:
    import boto3
    from moto import mock_dynamodb

    has_boto_libs = True
except ImportError:
    has_boto_libs = False


@pytest.fixture()
def kvstore_from_mocked_table(
    monkeypatch: MonkeyPatch,
) -> Generator[DynamoDBKVStore, None, None]:
    monkeypatch.setenv("MOTO_ALLOW_NONEXISTENT_REGION", "True")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "Andes")

    table_name = "test_table"
    with mock_dynamodb():
        client = boto3.client("dynamodb")
        client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "collection", "AttributeType": "S"},
                {"AttributeName": "key", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "collection", "KeyType": "HASH"},
                {"AttributeName": "key", "KeyType": "RANGE"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield DynamoDBKVStore.from_table_name(table_name)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_put_get(kvstore_from_mocked_table: DynamoDBKVStore) -> None:
    test_key = "test_key"
    test_value = {"test_str": "test_str", "test_float": 3.14}
    kvstore_from_mocked_table.put(key=test_key, val=test_value)
    item = kvstore_from_mocked_table.get(key=test_key)
    assert item == test_value


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_get_non_existent(kvstore_from_mocked_table: DynamoDBKVStore) -> None:
    test_key = "test_key"
    item = kvstore_from_mocked_table.get(key=test_key)
    assert item is None


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_put_get_multiple_collections(
    kvstore_from_mocked_table: DynamoDBKVStore,
) -> None:
    test_key = "test_key"
    test_item_collection_a = {"test_obj_key": "a"}
    test_item_collection_b = {"test_obj_key": "b"}
    kvstore_from_mocked_table.put(
        key=test_key, val=test_item_collection_a, collection="test_collection_a"
    )
    kvstore_from_mocked_table.put(
        key=test_key, val=test_item_collection_b, collection="test_collection_b"
    )
    item_collection_a = kvstore_from_mocked_table.get(
        key=test_key, collection="test_collection_a"
    )
    item_collection_b = kvstore_from_mocked_table.get(
        key=test_key, collection="test_collection_b"
    )
    assert test_item_collection_a == item_collection_a
    assert test_item_collection_b == item_collection_b


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_delete(kvstore_from_mocked_table: DynamoDBKVStore) -> None:
    test_key = "test_key"
    test_item = {"test_item": "test_item_val"}
    kvstore_from_mocked_table.put(key=test_key, val=test_item)
    item = kvstore_from_mocked_table.get(key=test_key)
    assert item == test_item
    assert kvstore_from_mocked_table.delete(key=test_key)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_delete_non_existent(kvstore_from_mocked_table: DynamoDBKVStore) -> None:
    test_key = "test_key"
    test_item = {"test_item_key": "test_item_val"}
    kvstore_from_mocked_table.put(key=test_key, val=test_item)
    assert kvstore_from_mocked_table.delete(key="wrong_key") is False


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_get_all(kvstore_from_mocked_table: DynamoDBKVStore) -> None:
    test_key_a = "test_key_a"
    test_item_a = {"test_item_key": "test_item_val_a"}

    test_key_b = "test_key_b"
    test_item_b = {"test_item_key": "test_item_val_b"}

    kvstore_from_mocked_table.put(key=test_key_a, val=test_item_a)
    kvstore_from_mocked_table.put(key=test_key_b, val=test_item_b)

    items = kvstore_from_mocked_table.get_all()
    assert items == {test_key_a: test_item_a, test_key_b: test_item_b}
