import pytest
from moto import mock_aws
import boto3
import time
from llama_index.storage.chat_store.dynamodb.base import DynamoDBChatStore
from llama_index.core.base.llms.types import ChatMessage


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
