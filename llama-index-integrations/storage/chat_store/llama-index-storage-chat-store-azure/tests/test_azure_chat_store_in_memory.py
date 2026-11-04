"""Behavioural tests for AzureChatStore against an in-memory fake table service."""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, TransactionOperation
from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.azure import AzureChatStore


class _AsyncIter:
    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncIter":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class FakeAsyncTableClient:
    """Minimal in-memory stand-in for azure.data.tables.aio.TableClient."""

    def __init__(self) -> None:
        self.rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def query_entities(self, query_filter: str) -> _AsyncIter:
        # Only supports the "PartitionKey eq '<key>'" filter used by the store.
        partition_key = query_filter.split("'")[1]
        rows = [
            dict(row)
            for (pk, _), row in sorted(self.rows.items())
            if pk == partition_key
        ]
        return _AsyncIter(rows)

    async def get_entity(self, partition_key: str, row_key: str) -> Dict[str, Any]:
        try:
            return dict(self.rows[(partition_key, row_key)])
        except KeyError:
            raise ResourceNotFoundError("Entity not found")

    async def create_entity(self, entity: Dict[str, Any]) -> None:
        pk_rk = (entity["PartitionKey"], entity["RowKey"])
        if pk_rk in self.rows:
            raise ResourceExistsError("Entity already exists")
        self.rows[pk_rk] = dict(entity)

    async def upsert_entity(self, entity: Dict[str, Any], mode: Any = None) -> None:
        pk_rk = (entity["PartitionKey"], entity["RowKey"])
        self.rows[pk_rk] = {**self.rows.get(pk_rk, {}), **entity}

    async def delete_entity(self, partition_key: str, row_key: str) -> None:
        # Azure Tables treats deleting a missing entity as a no-op.
        self.rows.pop((partition_key, row_key), None)

    async def submit_transaction(self, operations: Any) -> None:
        for op, entity in list(operations):
            if op == TransactionOperation.DELETE:
                await self.delete_entity(entity["PartitionKey"], entity["RowKey"])
            elif op == TransactionOperation.CREATE:
                await self.create_entity(entity)
            else:  # pragma: no cover - not used by the store
                raise NotImplementedError(op)


class FakeAsyncTableServiceClient:
    def __init__(self) -> None:
        self.tables: Dict[str, FakeAsyncTableClient] = {}

    async def create_table_if_not_exists(self, table_name: str) -> FakeAsyncTableClient:
        return self.tables.setdefault(table_name, FakeAsyncTableClient())


@pytest.fixture()
def store() -> AzureChatStore:
    return AzureChatStore(
        MagicMock(spec=TableServiceClient), FakeAsyncTableServiceClient()
    )


def _msgs(*contents: str) -> List[ChatMessage]:
    return [ChatMessage(role="user", content=c) for c in contents]


def _contents(messages: List[ChatMessage]) -> List[str]:
    return [m.content for m in messages]


def test_add_and_get_messages(store: AzureChatStore) -> None:
    for c in ("a", "b", "c"):
        store.add_message("k", ChatMessage(role="user", content=c))
    assert _contents(store.get_messages("k")) == ["a", "b", "c"]
    assert store.get_keys() == ["k"]


def test_add_message_at_index_inserts(store: AzureChatStore) -> None:
    store.set_messages("k", _msgs("a", "c"))
    store.add_message("k", ChatMessage(role="user", content="b"), idx=1)
    assert _contents(store.get_messages("k")) == ["a", "b", "c"]
    store.add_message("k", ChatMessage(role="user", content="d"))
    assert _contents(store.get_messages("k")) == ["a", "b", "c", "d"]


def test_delete_message_returns_message_and_keeps_order(store: AzureChatStore) -> None:
    store.set_messages("k", _msgs("a", "b", "c", "d"))

    deleted = store.delete_message("k", 1)
    assert deleted is not None and deleted.content == "b"
    assert _contents(store.get_messages("k")) == ["a", "c", "d"]

    # Indices refer to positions in the current list after a deletion.
    deleted = store.delete_message("k", 1)
    assert deleted is not None and deleted.content == "c"
    assert _contents(store.get_messages("k")) == ["a", "d"]

    # Appending after deleting from the middle must not leave gaps or overwrite.
    store.add_message("k", ChatMessage(role="user", content="e"))
    assert _contents(store.get_messages("k")) == ["a", "d", "e"]


def test_delete_message_out_of_range_or_missing_key(store: AzureChatStore) -> None:
    assert store.delete_message("missing", 0) is None
    store.set_messages("k", _msgs("a"))
    assert store.delete_message("k", 5) is None
    assert _contents(store.get_messages("k")) == ["a"]


def test_delete_last_message(store: AzureChatStore) -> None:
    store.set_messages("k", _msgs("a", "b"))

    deleted = store.delete_last_message("k")
    assert deleted is not None and deleted.content == "b"
    assert _contents(store.get_messages("k")) == ["a"]

    store.add_message("k", ChatMessage(role="user", content="c"))
    assert _contents(store.get_messages("k")) == ["a", "c"]

    assert store.delete_last_message("k").content == "c"
    assert store.delete_last_message("k").content == "a"
    assert store.delete_last_message("k") is None
    assert store.delete_last_message("missing") is None
    assert store.get_messages("k") == []


def test_delete_messages_returns_deleted(store: AzureChatStore) -> None:
    store.set_messages("k", _msgs("a", "b"))
    deleted = store.delete_messages("k")
    assert deleted is not None and _contents(deleted) == ["a", "b"]
    assert store.get_messages("k") == []
    store.add_message("k", ChatMessage(role="user", content="c"))
    assert _contents(store.get_messages("k")) == ["c"]


@pytest.mark.asyncio
async def test_async_delete_message(store: AzureChatStore) -> None:
    await store.aset_messages("k", _msgs("a", "b", "c"))
    deleted = await store.adelete_message("k", 0)
    assert deleted is not None and deleted.content == "a"
    assert _contents(await store.aget_messages("k")) == ["b", "c"]
    last = await store.adelete_last_message("k")
    assert last is not None and last.content == "c"
    assert _contents(await store.aget_messages("k")) == ["b"]
