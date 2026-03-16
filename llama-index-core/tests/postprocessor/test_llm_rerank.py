"""Test LLM reranker."""

import re
from typing import Any, List
from unittest.mock import patch

import pytest

from llama_index.core.base.llms.types import (
    ChatResponse,
    ChatMessage,
    TextBlock,
    AudioBlock,
    ImageBlock,
    CompletionResponse,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.schema import (
    BaseNode,
    Node,
    NodeWithScore,
    QueryBundle,
    TextNode,
    ImageNode,
    MediaResource,
)


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def mp3_bytes() -> bytes:
    """
    Small mp3 file bytes (0.2 seconds of audio).
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x06\x00\x00\x03<\x00YYYYYYYYYYYYYYYYzzzzzzzzzzzzzzzzz\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x03<\xa6\xbc`\x8e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4_\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp4_bytes() -> bytes:
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def mock_llm_chat_or_complete(
    self: Any, messages_or_formatted_prompt, **kwargs
) -> ChatResponse:
    """Patch llm predictor predict."""
    node_to_choice_and_score = {
        "Test": (True, "1"),
        "Test2": (False, "0"),
        "Test3": (True, "3"),
        "Test4": (False, "0"),
        "Test5": (True, "5"),
        "Test6": (False, "0"),
        "Test7": (True, "7"),
        "Test8": (False, "0"),
        "image": (True, "5"),
        "audio": (True, "7"),
        "video": (False, "0"),
    }
    doc_regx = r"Document (\d+):(?:\n(Test\d*)|\Z)"
    choices_and_scores = []
    doc = None
    if isinstance(messages_or_formatted_prompt, str):
        messages = [ChatMessage(role="user", content=messages_or_formatted_prompt)]
        chat_response = False
    else:
        messages = messages_or_formatted_prompt
        chat_response = True
    for message in messages:
        for block in message.blocks:
            if block.block_type == "text":
                for doc_num, content in re.findall(doc_regx, block.text):
                    doc = doc_num
                    if content:
                        choice, score = node_to_choice_and_score[content]
                        if choice:
                            choices_and_scores.append((int(doc), score))
            else:
                choice, score = node_to_choice_and_score[block.block_type]
                if choice:
                    choices_and_scores.append((int(doc), score))

    result_strs = [f"Doc: {c!s}, Relevance: {s}" for c, s in choices_and_scores]
    return (
        ChatResponse(
            message=ChatMessage(blocks=[TextBlock(text="\n".join(result_strs))])
        )
        if chat_response
        else CompletionResponse(text="\n".join(result_strs))
    )


def mock_format_node_batch_fn(nodes: List[BaseNode]) -> str:
    """Mock format node batch fn."""
    return "\n".join([node.get_content() for node in nodes])


@patch.object(
    MockLLM,
    "complete",
    mock_llm_chat_or_complete,
)
def test_llm_rerank() -> None:
    """Test LLM rerank."""
    nodes = [
        TextNode(text="Test"),
        TextNode(text="Test2"),
        TextNode(text="Test3"),
        TextNode(text="Test4"),
        TextNode(text="Test5"),
        TextNode(text="Test6"),
        TextNode(text="Test7"),
        TextNode(text="Test8"),
    ]
    nodes_with_score = [NodeWithScore(node=n) for n in nodes]

    # choice batch size 4 (so two batches)
    # take top-3 across all data
    llm_rerank = LLMRerank(choice_batch_size=4, top_n=3)
    query_str = "What is?"
    result_nodes = llm_rerank.postprocess_nodes(
        nodes_with_score, QueryBundle(query_str)
    )
    assert len(result_nodes) == 3
    assert result_nodes[0].node.get_content() == "Test7"
    assert result_nodes[1].node.get_content() == "Test5"
    assert result_nodes[2].node.get_content() == "Test3"


@patch.object(
    MockLLM,
    "chat",
    mock_llm_chat_or_complete,
)
def test_llm_rerank_multimodal(png_1px_b64, mp3_bytes, mp4_bytes) -> None:
    """Test LLM rerank."""
    nodes = [
        TextNode(text="Test"),
        TextNode(text="Test2"),
        TextNode(text="Test3"),
        TextNode(text="Test4"),
        ImageNode(image=png_1px_b64),
        Node(image_resource=MediaResource(data=png_1px_b64)),
        Node(audio_resource=MediaResource(data=mp3_bytes)),
        Node(video_resource=MediaResource(data=mp4_bytes)),
    ]
    nodes_with_score = [NodeWithScore(node=n) for n in nodes]

    # choice batch size 4 (so two batches)
    # take top-4 across all data
    llm_rerank_mm = LLMRerank(
        llm=MockLLM(is_chat_model=True),
        choice_batch_size=4,
        top_n=4,
    )

    query_str = "What is?"
    result_nodes = llm_rerank_mm.postprocess_nodes(
        nodes_with_score, QueryBundle(query_str)
    )
    assert len(result_nodes) == 4
    assert result_nodes[0].node.get_content_blocks() == [AudioBlock(audio=mp3_bytes)]
    assert result_nodes[1].node.get_content_blocks() == [ImageBlock(image=png_1px_b64)]
    assert result_nodes[2].node.get_content_blocks() == [ImageBlock(image=png_1px_b64)]
    assert result_nodes[3].node.get_content_blocks() == [TextBlock(text="Test3")]
