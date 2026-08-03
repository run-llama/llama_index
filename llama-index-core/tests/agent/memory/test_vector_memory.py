"""Test vector memory."""

import json
from typing import Any, List
from llama_index.core.memory import VectorMemory
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from unittest.mock import patch
from llama_index.core.llms import ChatMessage
from llama_index.core.vector_stores.utils import (
    _validate_is_flat_dict,
    node_to_metadata_dict,
)


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Jerry likes juice.":
        return [1, 1, 0, 0, 0]
    elif text == "Bob likes burgers.":
        return [0, 1, 0, 1, 0]
    elif text == "Alice likes apples.":
        return [0, 0, 1, 0, 0]
    elif text == "What does Jerry like?":
        return [1, 1, 0, 0, 1]
    elif (
        text == "Jerry likes juice. That's nice."
    ):  # vector memory batches conversation turns starting with user
        return [1, 1, 0, 0, 1]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_vector_memory(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """Test vector memory."""
    # arrange
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 1},
    )
    msgs = [
        ChatMessage.from_str("Jerry likes juice.", "user"),
        ChatMessage.from_str("That's nice.", "assistant"),
        ChatMessage.from_str("Bob likes burgers.", "user"),
        ChatMessage.from_str("Alice likes apples.", "user"),
    ]
    for m in msgs:
        vector_memory.put(m)

    # act
    msgs = vector_memory.get("What does Jerry like?")

    # assert
    assert len(msgs) == 2
    assert msgs[0].content == "Jerry likes juice."
    assert msgs[1].content == "That's nice."


def test_vector_memory_metadata_is_flat() -> None:
    """
    Regression test for https://github.com/run-llama/llama_index/issues/15681.

    `sub_dicts` used to be stored as a raw list of dicts in node metadata,
    which vector stores that require flat scalar metadata (e.g. Chroma, via
    `flat_metadata=True`) reject with a `ValueError`. It must now be a
    JSON-encoded string.
    """
    vector_memory = VectorMemory.from_defaults(
        vector_store=None, embed_model=MockEmbedding(embed_dim=5)
    )
    vector_memory.put(ChatMessage.from_str("Jerry likes juice.", "user"))
    vector_memory.put(ChatMessage.from_str("That's nice.", "assistant"))

    metadata = node_to_metadata_dict(
        vector_memory.cur_batch_textnode, remove_text=True, flat_metadata=True
    )
    # should not raise, unlike before the fix
    _validate_is_flat_dict(metadata)

    sub_dicts = json.loads(metadata["sub_dicts"])
    assert len(sub_dicts) == 2
    assert sub_dicts[0]["content"] == "Jerry likes juice."
    assert sub_dicts[1]["content"] == "That's nice."


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_vector_memory_get_with_legacy_list_sub_dicts(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """
    `get()` must still work against nodes persisted before the fix, where
    `sub_dicts` metadata was a raw list of dicts rather than a JSON string.
    """
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 1},
    )
    vector_memory.put(ChatMessage.from_str("Jerry likes juice.", "user"))
    vector_memory.put(ChatMessage.from_str("That's nice.", "assistant"))

    # simulate a node persisted by a pre-fix version of VectorMemory
    legacy_sub_dicts = json.loads(
        vector_memory.cur_batch_textnode.metadata["sub_dicts"]
    )
    vector_memory.cur_batch_textnode.metadata["sub_dicts"] = legacy_sub_dicts
    vector_memory._commit_node(override_last=True)

    msgs = vector_memory.get("What does Jerry like?")

    assert len(msgs) == 2
    assert msgs[0].content == "Jerry likes juice."
    assert msgs[1].content == "That's nice."
