import asyncio
import json
import pytest
from pydantic_core import PydanticSerializationError

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
async def test_concurrent_initialization_waits_for_data_restore(
    chat_store: SQLAlchemyChatStore, monkeypatch: pytest.MonkeyPatch
):
    chat_store._db_data = [
        {
            "key": "user1",
            "timestamp": 1,
            "role": "user",
            "status": MessageStatus.ACTIVE.value,
            "data": ChatMessage(role="user", content="restored").model_dump(
                mode="json"
            ),
        }
    ]
    tables_ready = asyncio.Event()
    finish_setup = asyncio.Event()
    setup_tables = SQLAlchemyChatStore._setup_tables

    async def paused_setup_tables(self, engine):
        table = await setup_tables(self, engine)
        tables_ready.set()
        await finish_setup.wait()
        return table

    monkeypatch.setattr(SQLAlchemyChatStore, "_setup_tables", paused_setup_tables)
    first = asyncio.create_task(chat_store._initialize())
    second = None
    try:
        await asyncio.wait_for(tables_ready.wait(), timeout=5)
        second = asyncio.create_task(chat_store._initialize())
        await asyncio.sleep(0)
        assert not second.done()
    finally:
        finish_setup.set()
        await asyncio.gather(first, *([second] if second is not None else []))

    assert first.result() == second.result()
    assert [m.content for m in await chat_store.get_messages("user1")] == ["restored"]
    assert await chat_store.count_messages("user1") == 1
    assert len(chat_store.model_dump()["db_data"]) == 1


@pytest.mark.asyncio
async def test_concurrent_first_reads(chat_store: SQLAlchemyChatStore):
    results = await asyncio.gather(
        *(chat_store.get_messages("user1") for _ in range(5)), return_exceptions=True
    )
    assert results == [[] for _ in range(5)]

    await chat_store.add_message("user1", ChatMessage(role="user", content="hello"))
    assert [m.content for m in await chat_store.get_messages("user1")] == ["hello"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_initialization_waiter_retries_interrupted_setup(
    chat_store: SQLAlchemyChatStore, monkeypatch: pytest.MonkeyPatch, cancel: bool
):
    chat_store._db_data = [
        {
            "key": "user1",
            "timestamp": 1,
            "role": "user",
            "status": MessageStatus.ACTIVE.value,
            "data": ChatMessage(role="user", content="restored").model_dump(
                mode="json"
            ),
        }
    ]
    tables_ready = asyncio.Event()
    interrupt_setup = asyncio.Event()
    setup_tables = SQLAlchemyChatStore._setup_tables
    attempts = 0

    async def interrupted_setup_tables(self, engine):
        nonlocal attempts
        attempts += 1
        table = await setup_tables(self, engine)
        if attempts == 1:
            tables_ready.set()
            await interrupt_setup.wait()
            raise RuntimeError("interrupted setup")
        return table

    monkeypatch.setattr(SQLAlchemyChatStore, "_setup_tables", interrupted_setup_tables)
    first = asyncio.create_task(chat_store._initialize())
    second = None
    try:
        await asyncio.wait_for(tables_ready.wait(), timeout=5)
        second = asyncio.create_task(chat_store._initialize())
        await asyncio.sleep(0)
        if cancel:
            first.cancel()
        else:
            interrupt_setup.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
    finally:
        interrupt_setup.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )

    assert isinstance(results[0], asyncio.CancelledError if cancel else RuntimeError)
    assert not isinstance(results[1], BaseException)
    assert attempts == 2
    assert [m.content for m in await chat_store.get_messages("user1")] == ["restored"]


@pytest.mark.asyncio
async def test_serialize_during_initialization(
    chat_store: SQLAlchemyChatStore, monkeypatch: pytest.MonkeyPatch
):
    tables_ready = asyncio.Event()
    finish_setup = asyncio.Event()
    setup_tables = SQLAlchemyChatStore._setup_tables

    async def paused_setup_tables(self, engine):
        table = await setup_tables(self, engine)
        tables_ready.set()
        await finish_setup.wait()
        return table

    monkeypatch.setattr(SQLAlchemyChatStore, "_setup_tables", paused_setup_tables)
    initializing = asyncio.create_task(chat_store._initialize())
    try:
        await asyncio.wait_for(tables_ready.wait(), timeout=5)
        with pytest.raises(
            PydanticSerializationError, match="initialization to finish"
        ):
            chat_store.model_dump()
    finally:
        finish_setup.set()
        await initializing

    assert chat_store.model_dump()["db_data"] == []


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
