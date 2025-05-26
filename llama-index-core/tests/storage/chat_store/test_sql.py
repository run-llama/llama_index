import json
import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.sql import (
    SQLAlchemyChatStore,
    MessageStatus,
)


@pytest.fixture()
def chat_store() -> SQLAlchemyChatStore:
    """Create a SQLAlchemyChatStore for testing."""
    return SQLAlchemyChatStore(
        table_name="test_messages",
        async_database_uri="sqlite+aiosqlite:///:memory:",
    )


@pytest.mark.asyncio
async def test_add_get_messages(chat_store: SQLAlchemyChatStore):
    """Test adding and retrieving messages."""
    # Add messages
    await chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    await chat_store.add_message(
        "user1", ChatMessage(role="assistant", content="world")
    )

    # Test getting messages
    messages = await chat_store.get_messages("user1")
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "world"

    # Test with non-existent key
    empty_messages = await chat_store.get_messages("nonexistent")
    assert len(empty_messages) == 0


@pytest.mark.asyncio
async def test_add_messages_batch(chat_store: SQLAlchemyChatStore):
    """Test adding messages in batch."""
    batch_messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="world"),
        ChatMessage(role="user", content="how are you?"),
    ]

    await chat_store.add_messages("batch_user", batch_messages)

    messages = await chat_store.get_messages("batch_user")
    assert len(messages) == 3
    assert [m.content for m in messages] == ["hello", "world", "how are you?"]


@pytest.mark.asyncio
async def test_count_messages(chat_store: SQLAlchemyChatStore):
    """Test counting messages."""
    batch_messages = [
        ChatMessage(role="user", content="message1"),
        ChatMessage(role="assistant", content="message2"),
        ChatMessage(role="user", content="message3"),
    ]

    await chat_store.add_messages("count_user", batch_messages)

    count = await chat_store.count_messages("count_user")
    assert count == 3

    # Test count with non-existent key
    empty_count = await chat_store.count_messages("nonexistent")
    assert empty_count == 0


@pytest.mark.asyncio
async def test_set_messages(chat_store: SQLAlchemyChatStore):
    """Test setting messages (replacing existing ones)."""
    # Add initial messages
    await chat_store.add_message(
        "replace_user", ChatMessage(role="user", content="initial")
    )

    # Replace with new set
    new_messages = [
        ChatMessage(role="user", content="replaced1"),
        ChatMessage(role="assistant", content="replaced2"),
    ]
    await chat_store.set_messages("replace_user", new_messages)

    # Verify replacement
    messages = await chat_store.get_messages("replace_user")
    assert len(messages) == 2
    assert [m.content for m in messages] == ["replaced1", "replaced2"]


@pytest.mark.asyncio
async def test_delete_message(chat_store: SQLAlchemyChatStore):
    """Test deleting a specific message."""
    batch_messages = [
        ChatMessage(role="user", content="message1"),
        ChatMessage(role="assistant", content="message2"),
        ChatMessage(role="user", content="message3"),
    ]

    await chat_store.add_messages("delete_user", batch_messages)

    # Get messages to find their IDs
    async with chat_store._async_session_factory() as session:
        result = await session.execute(
            chat_store._table.select().where(chat_store._table.c.key == "delete_user")
        )
        rows = result.fetchall()

    # Delete the middle message
    middle_id = rows[1].id
    deleted_message = await chat_store.delete_message("delete_user", middle_id)

    # Verify deletion
    assert deleted_message.content == "message2"

    remaining_messages = await chat_store.get_messages("delete_user")
    assert len(remaining_messages) == 2
    assert [m.content for m in remaining_messages] == ["message1", "message3"]


@pytest.mark.asyncio
async def test_delete_messages(chat_store: SQLAlchemyChatStore):
    """Test deleting all messages for a key."""
    # Add messages for multiple users
    await chat_store.add_message(
        "delete_all_user1", ChatMessage(role="user", content="user1_message")
    )
    await chat_store.add_message(
        "delete_all_user2", ChatMessage(role="user", content="user2_message")
    )

    # Delete messages for user1
    await chat_store.delete_messages("delete_all_user1")

    # Verify deletion
    user1_messages = await chat_store.get_messages("delete_all_user1")
    user2_messages = await chat_store.get_messages("delete_all_user2")

    assert len(user1_messages) == 0
    assert len(user2_messages) == 1


@pytest.mark.asyncio
async def test_delete_oldest_messages(chat_store: SQLAlchemyChatStore):
    """Test deleting oldest messages."""
    batch_messages = [
        ChatMessage(role="user", content="oldest"),
        ChatMessage(role="assistant", content="middle"),
        ChatMessage(role="user", content="newest"),
    ]

    await chat_store.add_messages("oldest_test", batch_messages)

    # Delete oldest message
    deleted = await chat_store.delete_oldest_messages("oldest_test", 1)

    # Verify deleted message
    assert len(deleted) == 1
    assert deleted[0].content == "oldest"

    # Verify remaining messages
    remaining = await chat_store.get_messages("oldest_test")
    assert len(remaining) == 2
    assert [m.content for m in remaining] == ["middle", "newest"]


@pytest.mark.asyncio
async def test_archive_oldest_messages(chat_store: SQLAlchemyChatStore):
    """Test archiving oldest messages."""
    batch_messages = [
        ChatMessage(role="user", content="oldest"),
        ChatMessage(role="assistant", content="middle"),
        ChatMessage(role="user", content="newest"),
    ]

    await chat_store.add_messages("archive_test", batch_messages)

    # Archive oldest message
    archived = await chat_store.archive_oldest_messages("archive_test", 1)

    # Verify archived message
    assert len(archived) == 1
    assert archived[0].content == "oldest"

    # Verify active messages
    active = await chat_store.get_messages("archive_test", status=MessageStatus.ACTIVE)
    assert len(active) == 2
    assert [m.content for m in active] == ["middle", "newest"]

    # Verify archived messages
    archived_msgs = await chat_store.get_messages(
        "archive_test", status=MessageStatus.ARCHIVED
    )
    assert len(archived_msgs) == 1
    assert archived_msgs[0].content == "oldest"


@pytest.mark.asyncio
async def test_get_messages_with_limit_offset(chat_store: SQLAlchemyChatStore):
    """Test getting messages with limit and offset."""
    batch_messages = [
        ChatMessage(role="user", content="message1"),
        ChatMessage(role="assistant", content="message2"),
        ChatMessage(role="user", content="message3"),
        ChatMessage(role="assistant", content="message4"),
        ChatMessage(role="user", content="message5"),
    ]

    await chat_store.add_messages("pagination_test", batch_messages)

    # Test with limit
    limited = await chat_store.get_messages("pagination_test", limit=2)
    assert len(limited) == 2
    assert [m.content for m in limited] == ["message1", "message2"]

    # Test with offset
    offset = await chat_store.get_messages("pagination_test", offset=2)
    assert len(offset) == 3
    assert [m.content for m in offset] == ["message3", "message4", "message5"]

    # Test with both limit and offset
    paginated = await chat_store.get_messages("pagination_test", limit=2, offset=1)
    assert len(paginated) == 2
    assert [m.content for m in paginated] == ["message2", "message3"]


@pytest.mark.asyncio
async def test_get_keys(chat_store: SQLAlchemyChatStore):
    """Test getting all unique keys."""
    # Add messages for multiple users
    await chat_store.add_message(
        "keys_user1", ChatMessage(role="user", content="user1_message")
    )
    await chat_store.add_message(
        "keys_user2", ChatMessage(role="user", content="user2_message")
    )
    await chat_store.add_message(
        "keys_user3", ChatMessage(role="user", content="user3_message")
    )

    # Get all keys
    keys = await chat_store.get_keys()

    # Verify keys (note: other tests may add more keys)
    expected_keys = {"keys_user1", "keys_user2", "keys_user3"}
    assert expected_keys.issubset(set(keys))


@pytest.mark.asyncio
async def test_dump_load_store(chat_store: SQLAlchemyChatStore):
    """Test dumping and loading the store."""
    # Add some messages
    await chat_store.add_message(
        "dump_user1", ChatMessage(role="user", content="message1")
    )
    await chat_store.add_message(
        "dump_user2", ChatMessage(role="user", content="message2")
    )

    # Dump the store
    store_dict = chat_store.model_dump()

    # ensure it's valid json
    _ = json.dumps(store_dict)

    # Load the store
    loaded_store = SQLAlchemyChatStore.model_validate(store_dict)

    # verify the loaded store is equivalent to the original store
    assert loaded_store.table_name == chat_store.table_name
    assert loaded_store.async_database_uri == chat_store.async_database_uri

    # verify the messages are the same
    messages = await loaded_store.get_messages("dump_user1")
    assert len(messages) == 1
    assert messages[0].content == "message1"

    messages = await loaded_store.get_messages("dump_user2")
    assert len(messages) == 1
    assert messages[0].content == "message2"
