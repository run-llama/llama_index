import os
from typing import Any

from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
import pytest

SKIP_GROK = os.environ.get("GROK_API_KEY") is None


class Answer(BaseModel):
    """A simple answer."""

    answer: str


# Define the models to test against
GROK: list[dict[str, Any]] = (
    [
        {
            "model": "grok-4-fast-non-reasoning",
            "config": {
                "api_base": "https://api.x.ai/v1",
                "is_chat_model": True,
                "is_function_calling_model": True,
            },
        },
        {
            "model": "grok-4-fast-reasoning",
            "config": {
                "api_base": "https://api.x.ai/v1",
                "is_chat_model": True,
                "is_function_calling_model": True,
            },
        },
        {
            "model": "grok-4-fast-non-reasoning",
            "config": {
                "api_base": "https://api.x.ai/v1",
                "is_chat_model": True,
                "is_function_calling_model": True,
                "should_use_structured_outputs": True,
            },
        },
        {
            "model": "grok-4-fast-reasoning",
            "config": {
                "api_base": "https://api.x.ai/v1",
                "is_chat_model": True,
                "is_function_calling_model": True,
                "should_use_structured_outputs": True,
            },
        },
    ]
    if not SKIP_GROK
    else []
)


@pytest.fixture(params=GROK)
def llm(request) -> OpenAILike:
    return OpenAILike(
        model=request.param["model"],
        api_key=os.environ["GROK_API_KEY"],
        **request.param.get("config", {}),
    )


@pytest.mark.skipif(SKIP_GROK, reason="GROK_API_KEY not set")
def test_complete(llm: OpenAILike) -> None:
    """Test both sync and async complete methods."""
    prompt = "What is the capital of Switzerland?"

    sync_response = llm.complete(prompt)
    assert sync_response is not None
    assert len(sync_response.text) > 0
    assert "bern" in sync_response.text.lower()


@pytest.mark.skipif(SKIP_GROK, reason="GROK_API_KEY not set")
def test_complete_structured(llm: OpenAILike) -> None:
    """Test both sync and async complete methods."""
    prompt = "What is the capital of Switzerland?"

    sync_response: Answer | None = llm.as_structured_llm(Answer).complete(prompt).raw
    assert sync_response is not None
    assert "bern" in sync_response.answer.lower()
