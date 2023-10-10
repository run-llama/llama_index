from typing import Any

import pytest
from llama_index.llms.base import ChatMessage
from pytest import MonkeyPatch

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore
from llama_index.llms.cohere import Cohere


def mock_completion_with_retry(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://docs.cohere.com/reference/generate
    return cohere.responses.Generations.from_dict(
        {
            "id": "21caa4c4-6b88-45f7-b144-14ef4985384c",
            "generations": [
                {
                    "id": "b5e2bb70-bc9c-4f86-a22e-5b5fd13a3482",
                    "text": "\n\nThis is indeed a test",
                    "finish_reason": "COMPLETE",
                }
            ],
            "prompt": "test prompt",
            "meta": {"api_version": {"version": "1"}},
        },
        return_likelihoods=False,
    )


async def mock_acompletion_with_retry(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://docs.cohere.com/reference/generate
    return cohere.responses.Generations.from_dict(
        {
            "id": "21caa4c4-6b88-45f7-b144-14ef4985384c",
            "generations": [
                {
                    "id": "b5e2bb70-bc9c-4f86-a22e-5b5fd13a3482",
                    "text": "\n\nThis is indeed a test",
                    "finish_reason": "COMPLETE",
                }
            ],
            "prompt": "test prompt",
            "meta": {"api_version": {"version": "1"}},
        },
        return_likelihoods=False,
    )


def mock_chat_with_retry(*args: Any, **kwargs: Any) -> dict:
    return cohere.responses.Chat.from_dict(
        {
            "chatlog": None,
            "citations": None,
            "conversation_id": None,
            "documents": None,
            "generation_id": "357d15b3-9bd4-4170-9439-2e4cef2242c8",
            "id": "25c3632f-2d2a-4e15-acbd-804b976d0568",
            "is_search_required": None,
            "message": "test prompt",
            "meta": {"api_version": {"version": "1"}},
            "preamble": None,
            "prompt": None,
            "response_id": "25c3632f-2d2a-4e15-acbd-804b976d0568",
            "search_queries": None,
            "search_results": None,
            "text": "\n\nThis is indeed a test",
            "token_count": {
                "billed_tokens": 66,
                "prompt_tokens": 64,
                "response_tokens": 9,
                "total_tokens": 73,
            },
        },
        client=None,
        message="test_prompt",
    )


async def mock_achat_with_retry(*args: Any, **kwargs: Any) -> dict:
    return cohere.responses.Chat.from_dict(
        {
            "chatlog": None,
            "citations": None,
            "conversation_id": None,
            "documents": None,
            "generation_id": "357d15b3-9bd4-4170-9439-2e4cef2242c8",
            "id": "25c3632f-2d2a-4e15-acbd-804b976d0568",
            "is_search_required": None,
            "message": "test prompt",
            "meta": {"api_version": {"version": "1"}},
            "preamble": None,
            "prompt": None,
            "response_id": "25c3632f-2d2a-4e15-acbd-804b976d0568",
            "search_queries": None,
            "search_results": None,
            "text": "\n\nThis is indeed a test",
            "token_count": {
                "billed_tokens": 66,
                "prompt_tokens": 64,
                "response_tokens": 9,
                "total_tokens": 73,
            },
        },
        client=None,
        message="test_prompt",
    )


@pytest.mark.skipif(cohere is None, reason="cohere not installed")
def test_completion_model_basic(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.cohere.completion_with_retry", mock_completion_with_retry
    )
    mock_api_key = "fake_key"
    llm = Cohere(model="command", api_key=mock_api_key)
    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert response.text == "\n\nThis is indeed a test"

    monkeypatch.setattr(
        "llama_index.llms.cohere.completion_with_retry", mock_chat_with_retry
    )

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.skipif(cohere is None, reason="cohere not installed")
@pytest.mark.asyncio()
async def test_async(monkeypatch: MonkeyPatch) -> None:
    mock_api_key = "fake_key"
    monkeypatch.setattr(
        "llama_index.llms.cohere.acompletion_with_retry", mock_acompletion_with_retry
    )
    llm = Cohere(model="command", api_key=mock_api_key)
    test_prompt = "test prompt"
    response = await llm.acomplete(test_prompt)
    assert response.text == "\n\nThis is indeed a test"

    monkeypatch.setattr(
        "llama_index.llms.cohere.acompletion_with_retry", mock_achat_with_retry
    )
    message = ChatMessage(role="user", content=test_prompt)
    chat_response = await llm.achat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"
