"""Tests for VectorMemory.get() sub_dict parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import VectorMemory


@dataclass
class _MockNode:
    metadata: Optional[Dict[str, Any]]


class _MockRetriever:
    def __init__(self, nodes: List[_MockNode]) -> None:
        self._nodes = nodes

    def retrieve(self, _query: str) -> List[_MockNode]:
        return self._nodes


def _make_memory_with_nodes(nodes: List[_MockNode]) -> VectorMemory:
    # We don't exercise indexing/embeddings here; we only need a valid VectorMemory instance.
    mem = VectorMemory.from_defaults(embed_model=MockEmbedding(embed_dim=1))
    mem.vector_index.as_retriever = lambda **_kwargs: _MockRetriever(nodes)  # type: ignore[method-assign]
    return mem


def test_vector_memory_get_mixed_nodes_with_and_without_sub_dicts() -> None:
    mem = _make_memory_with_nodes(
        [
            _MockNode(
                metadata={
                    "sub_dicts": [
                        ChatMessage.from_str("hello", "user").model_dump(),
                        ChatMessage.from_str("hi", "assistant").model_dump(),
                    ]
                }
            ),
            _MockNode(metadata={"source": "doc.txt"}),  # no sub_dicts
            _MockNode(metadata=None),  # missing metadata entirely
        ]
    )

    msgs = mem.get("query")

    assert [m.content for m in msgs] == ["hello", "hi"]


def test_vector_memory_get_nodes_with_only_document_metadata() -> None:
    mem = _make_memory_with_nodes(
        [
            _MockNode(metadata={"source": "doc1.txt", "page": 1}),
            _MockNode(metadata={"source": "doc2.txt", "page": 2}),
        ]
    )

    msgs = mem.get("query")

    assert msgs == []


def test_vector_memory_get_nodes_with_empty_sub_dicts() -> None:
    mem = _make_memory_with_nodes(
        [
            _MockNode(metadata={"sub_dicts": []}),
            _MockNode(metadata={"sub_dicts": []}),
        ]
    )

    msgs = mem.get("query")

    assert msgs == []
