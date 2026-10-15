from typing import List

import pytest
from fakeredis import FakeAsyncRedis, FakeRedis, FakeServer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.storage.chat_store.redis import RedisChatStore


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("decode_responses", [False, True], ids=["bytes", "str"])
@pytest.mark.parametrize(
    ("contents", "idx", "expected"),
    [
        pytest.param(["A", "B", "C"], 0, ["B", "C"], id="first"),
        pytest.param(["A", "B", "C"], 1, ["A", "C"], id="middle"),
        pytest.param(["A", "B", "C"], 2, ["A", "B"], id="last"),
        pytest.param(["A"], 0, [], id="only-message"),
        pytest.param(["A", "B", "C"], -1, ["A", "B", "C"], id="negative-index"),
        pytest.param(["A", "B", "C"], 3, ["A", "B", "C"], id="out-of-range"),
        pytest.param([], 0, [], id="missing-key"),
        pytest.param(["A", "B", "A"], 2, ["A", "B"], id="duplicate-content"),
    ],
)
async def test_delete_message(
    use_async: bool,
    decode_responses: bool,
    contents: List[str],
    idx: int,
    expected: List[str],
) -> None:
    server = FakeServer()
    key = "chat-history"
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=content,
            additional_kwargs={"metadata": {"source": "test"}},
        )
        for content in contents
    ]
    with FakeRedis(server=server, decode_responses=decode_responses) as redis_client:
        async with FakeAsyncRedis(
            server=server, decode_responses=decode_responses
        ) as aredis_client:
            store = RedisChatStore(
                redis_client=redis_client, aredis_client=aredis_client
            )
            if use_async:
                await store.aset_messages(key, messages)
                removed = await store.adelete_message(key, idx)
                remaining = await store.aget_messages(key)
            else:
                store.set_messages(key, messages)
                removed = store.delete_message(key, idx)
                remaining = store.get_messages(key)

            if 0 <= idx < len(messages):
                serialized = messages[idx].model_dump_json()
                assert removed == (
                    serialized if decode_responses else serialized.encode("utf-8")
                )
            else:
                assert removed is None
            assert [message.content for message in remaining] == expected
            assert all(
                message.role == MessageRole.USER
                and message.additional_kwargs == {"metadata": {"source": "test"}}
                for message in remaining
            )
            assert bool(redis_client.exists(key)) == bool(expected)
