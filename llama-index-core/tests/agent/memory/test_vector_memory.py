"""Test vector memory."""

from typing import Any, List
from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import VectorMemory
from llama_index.core.schema import TextNode


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


def test_vector_memory_get_handles_nodes_without_sub_dicts() -> None:
    """
    Test that VectorMemory.get() gracefully handles nodes without sub_dicts metadata.

    This occurs when using ChromaDB or other vector stores with a shared/pre-existing
    collection containing document chunks (e.g. from VectorStoreIndex.from_documents)
    that don't have the VectorMemory sub_dicts structure. Before the fix, this would
    raise KeyError: 'sub_dicts'.
    """
    # Create VectorMemory with in-memory store
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 5},
    )

    # Create mock nodes: one WITH sub_dicts (valid VectorMemory node),
    # one WITHOUT (e.g. from document indexing)
    valid_sub_dict = ChatMessage.from_str("hello from memory", "user").model_dump()
    node_with_sub_dicts = TextNode(
        text="User said hello",
        metadata={
            "sub_dicts": [valid_sub_dict],
        },
    )
    node_without_sub_dicts = TextNode(
        text="Some document content",
        metadata={"file_name": "doc.pdf", "page_label": "1"},  # No sub_dicts
    )

    # Mock the retriever to return both types (simulating shared Chroma collection)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        node_with_sub_dicts,
        node_without_sub_dicts,
    ]

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    vector_memory.vector_index = mock_index

    # Act - should not raise KeyError
    msgs = vector_memory.get("hello")

    # Assert - only messages from node with sub_dicts; node without is skipped
    assert len(msgs) == 1
    assert msgs[0].content == "hello from memory"
    assert msgs[0].role == MessageRole.USER


def test_vector_memory_get_handles_only_nodes_without_sub_dicts() -> None:
    """
    Test that VectorMemory.get() returns empty list when all nodes lack sub_dicts.

    When using a pre-populated Chroma collection (e.g. from document ingestion),
    retrieval may return only document nodes. Before the fix this raised KeyError.
    """
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 5},
    )

    # Only document-style nodes (no sub_dicts)
    document_nodes = [
        TextNode(
            text="Document chunk 1",
            metadata={"file_name": "doc1.pdf", "page_label": "1"},
        ),
        TextNode(
            text="Document chunk 2",
            metadata={"file_name": "doc2.pdf", "page_label": "2"},
        ),
    ]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = document_nodes

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    vector_memory.vector_index = mock_index

    # Act - should not raise KeyError, returns empty list
    msgs = vector_memory.get("query")

    assert len(msgs) == 0


def test_vector_memory_get_handles_nodes_with_empty_sub_dicts() -> None:
    """Test that VectorMemory.get() handles nodes with sub_dicts=[] (empty list)."""
    embed_model = MockEmbedding(embed_dim=5)
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 5},
    )

    node_with_empty_sub_dicts = TextNode(
        text="",
        metadata={"sub_dicts": []},
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [node_with_empty_sub_dicts]

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    vector_memory.vector_index = mock_index

    msgs = vector_memory.get("query")

    assert len(msgs) == 0
