import asyncio
from unittest import mock

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.maritalk import Maritalk


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in Maritalk.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_stream_chat_preserves_data_prefix_substring():
    """
    Text containing the literal 'data: ' must not be corrupted.

    Regression: the SSE prefix was stripped with ``line.replace(b"data: ", b"")``,
    which removes every occurrence, so a delta like 'see data: here' became
    'see here'. Only the leading 'data: ' prefix should be removed.
    """
    lines = [
        b'data: {"text":"see data: here"}',
        b'data: {"text":"!"}',
    ]
    mock_response = mock.Mock()
    mock_response.ok = True
    mock_response.iter_lines.return_value = iter(lines)

    llm = Maritalk(api_key="test-key")
    with mock.patch("requests.post", return_value=mock_response):
        responses = list(
            llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")])
        )

    assert [r.delta for r in responses] == ["see data: here", "!"]
    assert responses[-1].message.content == "see data: here!"


def test_astream_chat_preserves_data_prefix_substring():
    """Async counterpart of the streaming 'data: ' corruption regression."""
    lines = [
        'data: {"text":"see data: here"}',
        'data: {"text":"!"}',
    ]

    class _MockStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *args, **kwargs):
            return _MockStream()

    async def run():
        with mock.patch("httpx.AsyncClient", return_value=_MockAsyncClient()):
            gen = await Maritalk(api_key="test-key").astream_chat(
                [ChatMessage(role=MessageRole.USER, content="hi")]
            )
            return [r.delta async for r in gen]

    assert asyncio.run(run()) == ["see data: here", "!"]
