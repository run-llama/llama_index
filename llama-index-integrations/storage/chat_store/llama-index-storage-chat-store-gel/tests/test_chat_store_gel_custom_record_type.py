"""GelChatStore must use the configured ``record_type`` in every query."""

import os
import re
from typing import Any, List

import pytest
from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.gel import GelChatStore


class _RecordingClient:
    """Fake Gel client that records the queries it receives."""

    def __init__(self) -> None:
        self.queries: List[str] = []

    def query(self, query: str, **kwargs: Any) -> List[Any]:
        self.queries.append(query)
        return []

    def query_single(self, query: str, **kwargs: Any) -> Any:
        self.queries.append(query)
        return None


class _AsyncRecordingClient(_RecordingClient):
    async def query(self, query: str, **kwargs: Any) -> List[Any]:  # type: ignore[override]
        return _RecordingClient.query(self, query, **kwargs)

    async def query_single(self, query: str, **kwargs: Any) -> Any:  # type: ignore[override]
        return _RecordingClient.query_single(self, query, **kwargs)


def _exercise_sync(store: GelChatStore) -> None:
    message = ChatMessage(role="user", content="hi")
    store.set_messages("k", [message])
    store.get_messages("k")
    store.add_message("k", message)
    store.delete_message("k", 0)
    store.delete_last_message("k")
    store.delete_messages("k")
    store.get_keys()


def _assert_only_custom_type(queries: List[str]) -> None:
    assert len(queries) == 7
    for query in queries:
        assert "ChatRecord" in query, query
        assert re.search(r"\bRecord\b", query) is None, query


def test_queries_use_custom_record_type() -> None:
    store = GelChatStore(record_type="ChatRecord")
    client = _RecordingClient()
    store._sync_client = client  # bypass connection setup
    _exercise_sync(store)
    _assert_only_custom_type(client.queries)


@pytest.mark.asyncio
async def test_async_queries_use_custom_record_type() -> None:
    store = GelChatStore(record_type="ChatRecord")
    client = _AsyncRecordingClient()
    store._async_client = client  # bypass connection setup
    message = ChatMessage(role="user", content="hi")
    await store.aset_messages("k", [message])
    await store.aget_messages("k")
    await store.async_add_message("k", message)
    await store.adelete_message("k", 0)
    await store.adelete_last_message("k")
    await store.adelete_messages("k")
    await store.aget_keys()
    _assert_only_custom_type(client.queries)


def test_invalid_record_type_rejected() -> None:
    with pytest.raises(ValueError):
        GelChatStore(record_type="Record; delete Record")


@pytest.mark.skipif(
    os.environ.get("GEL_CUSTOM_RECORD_TYPE_TEST") is None,
    reason="requires a Gel project whose schema defines a ChatRecord type",
)
def test_custom_record_type_against_gel() -> None:
    store = GelChatStore(record_type="ChatRecord")
    try:
        store.set_messages("k", [ChatMessage(role="user", content="a")])
        store.add_message("k", ChatMessage(role="assistant", content="b"))
        assert [m.content for m in store.get_messages("k")] == ["a", "b"]
        assert store.get_keys() == ["k"]
        assert store.delete_last_message("k").content == "b"
        assert store.delete_message("k", 0).content == "a"
    finally:
        store.delete_messages("k")
