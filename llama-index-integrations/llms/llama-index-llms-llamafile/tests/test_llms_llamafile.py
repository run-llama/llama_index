from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.llamafile import Llamafile
from pytest import MonkeyPatch
import httpx
from json import dumps as json_dumps
import io
from collections import deque
from contextlib import contextmanager
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)


BASE_URL = "http://llamafile-host:8080"
DEFAULT_GENERATION_OPTS = {"temperature": 0.8, "seed": 0}


def test_class_extends_base_llm():
    names_of_base_classes = [b.__name__ for b in Llamafile.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_init():
    llm = Llamafile(temperature=0.1)
    assert llm.temperature == 0.1
    assert llm.seed == 0


def mock_completion_response() -> httpx.Response:
    content = json_dumps({"content": "the quick brown fox"})
    return httpx.Response(
        status_code=200,
        content=content,
        request=httpx.Request(method="POST", url=BASE_URL),
    )


@contextmanager
def mock_completion_response_stream():  # type: ignore[no-untyped-def]
    mock_response_content = deque(
        [
            'data: {"content":"the","multimodal":false,"slot_id":0,"stop":false}\n\n',
            'data: {"content":" quick","multimodal":false,"slot_id":0,"stop":false}\n\n',
        ]
    )

    def iter_lines(chunk_size=None):
        yield from mock_response_content

    response = httpx.Response(
        status_code=200,
        request=httpx.Request(method="POST", url=BASE_URL),
    )
    response.iter_lines = iter_lines

    try:
        yield response
    finally:
        response.close()


def mock_chat_response():
    content = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "the quick brown fox", "role": "assistant"},
            }
        ],
        "created": 1709747825,
        "id": "chatcmpl-mK93T0egYqsADjbBe5QSby8T3yx3kTWR",
        "model": "unknown",
        "object": "chat.completion",
        "usage": {"completion_tokens": 265, "prompt_tokens": 37, "total_tokens": 302},
    }
    content = json_dumps(content)
    return httpx.Response(
        status_code=200,
        content=content,
        request=httpx.Request(method="POST", url=BASE_URL),
    )


@contextmanager
def mock_chat_response_stream():
    mock_response_content = [
        'data: {"choices":[{"delta":{"content":"the"},"finish_reason":null,"index":0}],"created":1709748404,"id":"chatcmpl-xAHyuhQ8x4Ke9FVFrkiBSv2Je2xPmTj6","model":"unknown","object":"chat.completion.chunk"}',
        'data: {"choices":[{"delta":{"content":" quick"},"finish_reason":null,"index":0}],"created":1709748404,"id":"chatcmpl-xAHyuhQ8x4Ke9FVFrkiBSv2Je2xPmTj6","model":"unknown","object":"chat.completion.chunk"}',
    ]

    def iter_lines(chunk_size=None):
        yield from mock_response_content

    response = httpx.Response(
        status_code=200,
        request=httpx.Request(method="POST", url=BASE_URL),
    )
    response.iter_lines = iter_lines

    try:
        yield response
    finally:
        response.close()


def test_complete(monkeypatch: MonkeyPatch) -> None:
    llm = Llamafile(base_url=BASE_URL)

    def mock_post(self, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "prompt": "Test prompt",
            "stream": False,
            **DEFAULT_GENERATION_OPTS,
        }
        return mock_completion_response()

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    out = llm.complete("Test prompt")
    assert str(out) == "the quick brown fox"


def test_complete_with_kwargs(monkeypatch: MonkeyPatch) -> None:
    """Test generation options passed as kwargs to 'complete' override their default values."""
    llm = Llamafile(base_url=BASE_URL)

    def mock_post(self, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        generation_opts = dict(DEFAULT_GENERATION_OPTS)
        generation_opts["seed"] = 1
        assert json == {"prompt": "Test prompt", "stream": False, **generation_opts}
        return mock_completion_response()

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    out = llm.complete("Test prompt", seed=1)
    assert str(out) == "the quick brown fox"


def test_stream_complete(monkeypatch: MonkeyPatch) -> None:
    llm = Llamafile(base_url=BASE_URL)

    def mock_post(self, method, url, headers, json):  # type: ignore[no-untyped-def]
        assert method == "POST"
        assert url == f"{BASE_URL}/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "prompt": "Test prompt",
            "stream": True,
            **DEFAULT_GENERATION_OPTS,
        }
        return mock_completion_response_stream()

    monkeypatch.setattr(httpx.Client, "stream", mock_post)
    buff = io.StringIO()
    for chunk in llm.stream_complete("Test prompt"):
        buff.write(chunk.delta)

    assert buff.getvalue() == "the quick"


def test_chat(monkeypatch: MonkeyPatch) -> None:
    llm = Llamafile(base_url=BASE_URL)

    def mock_post(self, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/v1/chat/completions"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "messages": [{"role": "user", "content": "Test prompt"}],
            "options": DEFAULT_GENERATION_OPTS,
            "stream": False,
        }
        return mock_chat_response()

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    messages = [ChatMessage(role="user", content="Test prompt")]
    out: ChatResponse = llm.chat(messages)
    assert out.message.content == "the quick brown fox"


def test_stream_chat(monkeypatch: MonkeyPatch) -> None:
    llm = Llamafile(base_url=BASE_URL)

    def mock_post(self, method, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/v1/chat/completions"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "messages": [{"role": "user", "content": "Test prompt"}],
            "options": DEFAULT_GENERATION_OPTS,
            "stream": True,
        }
        return mock_chat_response_stream()

    monkeypatch.setattr(httpx.Client, "stream", mock_post)

    messages = [ChatMessage(role="user", content="Test prompt")]

    buff = io.StringIO()
    for chunk in llm.stream_chat(messages):
        buff.write(chunk.delta)

    assert buff.getvalue() == "the quick"
