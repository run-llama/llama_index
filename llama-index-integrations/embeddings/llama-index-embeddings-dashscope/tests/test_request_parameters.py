from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import dashscope
import pytest

from llama_index.embeddings.dashscope import DashScopeEmbedding


@pytest.mark.parametrize(
    ("method", "text_type", "expected_type"),
    [
        ("get_query_embedding", "document", "query"),
        ("get_text_embedding", "document", "document"),
        ("get_text_embedding", "query", "query"),
        ("get_text_embedding_batch", "document", "document"),
    ],
)
def test_text_embedding_forwards_text_type(
    monkeypatch: pytest.MonkeyPatch, method: str, text_type: str, expected_type: str
) -> None:
    call = Mock(
        return_value=SimpleNamespace(
            status_code=HTTPStatus.OK,
            output={"embeddings": [{"text_index": 0, "embedding": [0.1, 0.2]}]},
        )
    )
    monkeypatch.setattr(dashscope.TextEmbedding, "call", call)
    embedder = DashScopeEmbedding(api_key="placeholder-key", text_type=text_type)
    is_batch = method.endswith("batch")

    result = getattr(embedder, method)(["hello"] if is_batch else "hello")

    assert result == ([[0.1, 0.2]] if is_batch else [0.1, 0.2])
    call.assert_called_once_with(
        model=embedder.model_name,
        input=["hello"],
        api_key="placeholder-key",
        text_type=expected_type,
    )


def test_batch_text_embedding_forwards_text_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = Mock(
        return_value=SimpleNamespace(
            status_code=HTTPStatus.OK,
            output={"url": "https://example.com/output.jsonl"},
        )
    )
    monkeypatch.setattr(dashscope.BatchTextEmbedding, "call", call)
    embedder = DashScopeEmbedding(api_key="placeholder-key", text_type="query")

    assert embedder.get_batch_text_embedding("https://example.com/input.txt") == (
        "https://example.com/output.jsonl"
    )
    call.assert_called_once_with(
        model=embedder.model_name,
        url="https://example.com/input.txt",
        api_key="placeholder-key",
        text_type="query",
    )


@pytest.mark.parametrize("auto_truncation", [True, False])
def test_multimodal_embedding_forwards_auto_truncation(
    monkeypatch: pytest.MonkeyPatch, auto_truncation: bool
) -> None:
    call = Mock(
        return_value=SimpleNamespace(
            status_code=HTTPStatus.OK, output={"embedding": [0.1, 0.2]}
        )
    )
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", call)
    embedder = DashScopeEmbedding(api_key="placeholder-key")
    content = [{"text": "hello"}]

    assert embedder.get_multimodal_embedding(content, auto_truncation) == [0.1, 0.2]
    call.assert_called_once_with(
        model=embedder.model_name,
        input=content,
        api_key="placeholder-key",
        auto_truncation=auto_truncation,
    )
