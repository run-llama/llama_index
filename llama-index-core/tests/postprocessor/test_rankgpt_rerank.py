from typing import Any
from unittest.mock import patch
import asyncio

import pytest
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.schema import TextNode, NodeWithScore


def mock_rankgpt_chat(self: Any, messages, **kwargs: Any) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.SYSTEM, content="[2] > [1] > [3]")
    )


async def mock_rankgpt_achat(self, messages, **kwargs: Any) -> ChatResponse:
    # Mock api call
    await asyncio.sleep(1)
    return ChatResponse(
        message=ChatMessage(role=MessageRole.SYSTEM, content="[2] > [1] > [3]")
    )


nodes = [
    TextNode(text="Test"),
    TextNode(text="Test2"),
    TextNode(text="Test3"),
]
nodes_with_score = [NodeWithScore(node=n) for n in nodes]


@patch.object(
    MockLLM,
    "chat",
    mock_rankgpt_chat,
)
def test_rankgpt_rerank():
    rankgpt_rerank = RankGPTRerank(
        top_n=2,
        llm=MockLLM(),
    )
    result = rankgpt_rerank.postprocess_nodes(nodes_with_score, query_str="Test query")
    assert len(result) == 2
    assert result[0].node.get_content() == "Test2"
    assert result[1].node.get_content() == "Test"


@patch.object(
    MockLLM,
    "achat",
    mock_rankgpt_achat,
)
@pytest.mark.asyncio
async def test_rankgpt_rerank_async():
    rankgpt_rerank = RankGPTRerank(
        top_n=2,
        llm=MockLLM(),
    )
    result = await rankgpt_rerank.apostprocess_nodes(
        nodes_with_score, query_str="Test query"
    )
    assert len(result) == 2
    assert result[0].node.get_content() == "Test2"
    assert result[1].node.get_content() == "Test"
