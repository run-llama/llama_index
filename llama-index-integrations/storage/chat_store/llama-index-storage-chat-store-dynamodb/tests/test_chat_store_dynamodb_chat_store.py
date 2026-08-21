import asyncio
import inspect
import time

import boto3
import pytest
from moto import mock_aws

from llama_index.core.base.llms.types import ChatMessage
from llama_index.storage.chat_store.dynamodb.base import DynamoDBChatStore


class FakeAsyncDynamoDBTable:
    def __init__(self, primary_key: str) -> None:
        self.primary_key = primary_key
        self.items = {}

    async def put_item(self, Item):
        self.items[Item[self.primary_key]] = Item

    async def get_item(self, Key):
        key = Key[self.primary_key]
        if key not in self.items:
            return {}
        return {"Item": self.items[key]}

    async def delete_item(self, Key):
        self.items.pop(Key[self.primary_key], None)

    async def scan(self, ProjectionExpression, ExclusiveStartKey=None):
        keys = sorted(self.items)
        if ExclusiveStartKey is None and len(keys) > 1:
            return {
                "Items": [{ProjectionExpression: keys[0]}],
                "LastEvaluatedKey": {self.primary_key: keys[0]},
            }

        if ExclusiveStartKey is None:
            page_keys = keys
        else:
            last_key = ExclusiveStartKey[self.primary_key]
            page_keys = keys[keys.index(last_key) + 1 :]

        return {"Items": [{ProjectionExpression: key} for key in page_keys]}


@pytest.fixture()
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture()
def dynamo_db(aws_credentials):
    with mock_aws():
        yield boto3.resource("dynamodb", region_name="us-east-1")


@pytest.fixture()
def chat_store(dynamo_db):
    dynamo_db.create_table(
        TableName="TestTable",
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
    )
    return DynamoDBChatStore(table_name="TestTable", region_name="us-east-1")


@pytest.fixture()
def chat_store_with_ttl(dynamo_db):
    dynamo_db.create_table(
        TableName="TestTableTTL",
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
    )
    return DynamoDBChatStore(
        table_name="TestTableTTL",
        region_name="us-east-1",
        ttl_seconds=3600,  # 1 hour TTL
    )


@pytest.fixture()
def chat_store_with_custom_ttl(dynamo_db):
    dynamo_db.create_table(
        TableName="TestTableCustomTTL",
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
    )
    return DynamoDBChatStore(
        table_name="TestTableCustomTTL",
        region_name="us-east-1",
        ttl_seconds=7200,  # 2 hour TTL
        ttl_attribute="ExpiresAt",
    )


def test_set_get_messages(chat_store):
    messages = [ChatMessage(content="Hello"), ChatMessage(content="World")]
    chat_store.set_messages("TestSession", messages)
    retrieved_messages = chat_store.get_messages("TestSession")
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "Hello"
    assert retrieved_messages[1].content == "World"


def test_add_message(chat_store):
    initial_message = ChatMessage(content="Initial")
    chat_store.add_message("TestSession", initial_message)
    added_message = ChatMessage(content="Added")
    chat_store.add_message("TestSession", added_message)
    messages = chat_store.get_messages("TestSession")
    assert len(messages) == 2
    assert messages[1].content == "Added"


def test_delete_messages(chat_store):
    messages = [ChatMessage(content="Hello"), ChatMessage(content="World")]
    chat_store.set_messages("TestSession", messages)
    deleted_messages = chat_store.delete_messages("TestSession")
    assert len(deleted_messages) == 2
    assert deleted_messages[0].content == "Hello"
    assert chat_store.get_messages("TestSession") == []


def test_delete_message(chat_store):
    messages = [ChatMessage(content="First"), ChatMessage(content="Second")]
    chat_store.set_messages("TestSession", messages)
    chat_store.delete_message("TestSession", 0)
    remaining_messages = chat_store.get_messages("TestSession")
    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "Second"


def test_delete_last_message(chat_store):
    messages = [ChatMessage(content="First"), ChatMessage(content="Second")]
    chat_store.set_messages("TestSession", messages)
    last_message = chat_store.delete_last_message("TestSession")
    assert last_message.content == "Second"
    remaining_messages = chat_store.get_messages("TestSession")
    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First"


def test_get_keys(chat_store):
    chat_store._table.put_item(Item={"SessionId": "1", "History": []})
    chat_store._table.put_item(Item={"SessionId": "2", "History": []})
    keys = chat_store.get_keys()
    assert len(keys) == 2
    assert "1" in keys
    assert "2" in keys


def test_async_table_defaults_to_none(chat_store):
    assert chat_store._atable is None


def test_async_methods_initialize_table_and_return_values(chat_store, monkeypatch):
    table = FakeAsyncDynamoDBTable(primary_key=chat_store.primary_key)
    init_calls = 0

    async def init_async_table(self):
        nonlocal init_calls
        init_calls += 1
        self._atable = table

    monkeypatch.setattr(DynamoDBChatStore, "init_async_table", init_async_table)

    async def run_test():
        await chat_store.aset_messages(
            "TestSession",
            [ChatMessage(content="Hello"), ChatMessage(content="World")],
        )
        retrieved_messages = await chat_store.aget_messages("TestSession")
        assert [message.content for message in retrieved_messages] == [
            "Hello",
            "World",
        ]

        await chat_store.async_add_message("TestSession", ChatMessage(content="Added"))
        retrieved_messages = await chat_store.aget_messages("TestSession")
        assert [message.content for message in retrieved_messages] == [
            "Hello",
            "World",
            "Added",
        ]

        deleted_message = await chat_store.adelete_message("TestSession", 1)
        assert deleted_message is not None
        assert deleted_message.content == "World"

        last_message = await chat_store.adelete_last_message("TestSession")
        assert not inspect.iscoroutine(last_message)
        assert last_message is not None
        assert last_message.content == "Added"

        deleted_messages = await chat_store.adelete_messages("TestSession")
        assert deleted_messages is not None
        assert [message.content for message in deleted_messages] == ["Hello"]
        assert await chat_store.aget_messages("TestSession") == []

        await chat_store.aset_messages("KeyA", [])
        await chat_store.aset_messages("KeyB", [])
        assert await chat_store.aget_keys() == ["KeyA", "KeyB"]

    asyncio.run(run_test())
    assert init_calls > 0


def test_ttl_set_messages(chat_store_with_ttl):
    """Test that TTL is set when setting messages."""
    messages = [ChatMessage(content="Hello TTL")]
    chat_store_with_ttl.set_messages("TTLSession", messages)

    # Get the raw item from DynamoDB to check TTL
    response = chat_store_with_ttl._table.get_item(Key={"SessionId": "TTLSession"})
    item = response["Item"]

    # Verify TTL attribute exists
    assert "TTL" in item

    # Verify TTL is set to a future time (now + 3600 seconds)
    current_time = int(time.time())
    assert item["TTL"] > current_time
    # Allow some buffer for test execution time
    assert item["TTL"] <= current_time + 3700


def test_ttl_add_message(chat_store_with_ttl):
    """Test that TTL is set when adding a message."""
    message = ChatMessage(content="Hello TTL Add")
    chat_store_with_ttl.add_message("TTLAddSession", message)

    # Get the raw item from DynamoDB to check TTL
    response = chat_store_with_ttl._table.get_item(Key={"SessionId": "TTLAddSession"})
    item = response["Item"]

    # Verify TTL attribute exists
    assert "TTL" in item

    # Verify TTL is set to a future time (now + 3600 seconds)
    current_time = int(time.time())
    assert item["TTL"] > current_time
    # Allow some buffer for test execution time
    assert item["TTL"] <= current_time + 3700


def test_custom_ttl_attribute(chat_store_with_custom_ttl):
    """Test that custom TTL attribute name is used."""
    messages = [ChatMessage(content="Custom TTL Attribute")]
    chat_store_with_custom_ttl.set_messages("CustomTTLSession", messages)

    # Get the raw item from DynamoDB to check TTL
    response = chat_store_with_custom_ttl._table.get_item(
        Key={"SessionId": "CustomTTLSession"}
    )
    item = response["Item"]

    # Verify custom TTL attribute exists
    assert "ExpiresAt" in item
    assert "TTL" not in item

    # Verify TTL is set to a future time (now + 7200 seconds)
    current_time = int(time.time())
    assert item["ExpiresAt"] > current_time
    # Allow some buffer for test execution time
    assert item["ExpiresAt"] <= current_time + 7300


def test_no_ttl_when_disabled(chat_store):
    """Test that TTL is not set when ttl_seconds is None."""
    messages = [ChatMessage(content="No TTL")]
    chat_store.set_messages("NoTTLSession", messages)

    # Get the raw item from DynamoDB to check TTL
    response = chat_store._table.get_item(Key={"SessionId": "NoTTLSession"})
    item = response["Item"]

    # Verify TTL attribute does not exist
    assert "TTL" not in item
