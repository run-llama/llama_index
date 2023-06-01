from typing import Generator

import pytest
from pytest import MonkeyPatch

from llama_index.data_structs.data_structs import IndexGraph
from llama_index.storage.index_store.dynamodb_index_store import DynamoDBIndexStore
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


@pytest.fixture()
def ddb_index_store(kvstore_from_mocked_table: DynamoDBKVStore) -> DynamoDBIndexStore:
    return DynamoDBIndexStore(dynamodb_kvstore=kvstore_from_mocked_table)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_dynamodb_index_store(ddb_index_store: DynamoDBIndexStore) -> None:
    index_store = ddb_index_store

    index_struct = IndexGraph()
    index_store.add_index_struct(index_struct=index_struct)

    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
