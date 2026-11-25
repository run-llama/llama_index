from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.evaluation.multi_modal.faithfulness import (
    MultiModalFaithfulnessEvaluator,
)
from llama_index.core.evaluation.multi_modal.relevancy import (
    MultiModalRelevancyEvaluator,
)
from llama_index.core.llms.llm import LLM


def _make_mock_multi_modal_llm(reply_text: str = "Yes\nFeedback: relevant") -> MagicMock:
    llm = MagicMock(spec=LLM)
    chat_resp = ChatResponse(message=ChatMessage(role="assistant", content=reply_text))
    llm.chat = MagicMock(return_value=chat_resp)
    llm.achat = AsyncMock(return_value=chat_resp)
    return llm


@pytest.mark.asyncio
async def test_multi_modal_relevancy_aevaluate_uses_achat_and_preserves_contexts() -> None:
    llm = _make_mock_multi_modal_llm("Yes\nThe answer is relevant.")
    evaluator = MultiModalRelevancyEvaluator(multi_modal_llm=llm)

    sync_result = evaluator.evaluate(
        query="What is in the image?",
        response="A cat sitting on a mat.",
        contexts=["An image showing a cat sitting on a mat."],
    )
    assert llm.chat.call_count == 1
    assert llm.achat.call_count == 0
    assert sync_result.passing is True
    assert sync_result.score == 1.0
    assert sync_result.query == "What is in the image?"
    assert sync_result.contexts == ["An image showing a cat sitting on a mat."]

    async_result = await evaluator.aevaluate(
        query="What is in the image?",
        response="A cat sitting on a mat.",
        contexts=["An image showing a cat sitting on a mat."],
    )
    assert llm.chat.call_count == 1
    assert llm.achat.await_count == 1
    assert async_result == sync_result


@pytest.mark.asyncio
async def test_multi_modal_faithfulness_preserves_query_in_result() -> None:
    llm = _make_mock_multi_modal_llm("Yes\nThe response is faithful.")
    evaluator = MultiModalFaithfulnessEvaluator(multi_modal_llm=llm)

    sync_result = evaluator.evaluate(
        query="What is in the image?",
        response="A cat sitting on a mat.",
        contexts=["An image showing a cat sitting on a mat."],
    )
    assert sync_result.passing is True
    assert sync_result.score == 1.0
    assert sync_result.query == "What is in the image?"
    assert sync_result.contexts == ["An image showing a cat sitting on a mat."]

    async_result = await evaluator.aevaluate(
        query="What is in the image?",
        response="A cat sitting on a mat.",
        contexts=["An image showing a cat sitting on a mat."],
    )
    assert llm.achat.await_count == 1
    assert async_result == sync_result
