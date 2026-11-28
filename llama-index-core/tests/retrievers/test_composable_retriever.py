from typing import List

import pytest

from llama_index.core import MockEmbedding, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices import SummaryIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import (
    Document,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)


class OrthogonalEmbedding(BaseEmbedding):
    """Return orthogonal document and query vectors for deterministic zero scores."""

    def __init__(self) -> None:
        super().__init__(embed_dim=2)

    @classmethod
    def class_name(cls) -> str:
        return "OrthogonalEmbedding"

    def _get_text_embedding(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return [0.0, 1.0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


class ZeroScoreIndexNodeRetriever(BaseRetriever):
    """Return an index node whose score is exactly zero."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=IndexNode(text="child link", index_id="child"), score=0.0
            )
        ]


def test_composable_retrieval() -> None:
    """Test composable retrieval."""
    text_node = TextNode(text="This is a test text node.", id_="test_text_node")
    index_node = IndexNode(
        text="This is a test index node.",
        id_="test_index_node",
        index_id="test_index_node_index",
        obj=TextNode(text="Hidden node!", id_="hidden_node"),
    )

    index = SummaryIndex(nodes=[text_node, text_node], objects=[index_node])

    # Test retrieval
    retriever = index.as_retriever()
    nodes = retriever.retrieve("test")

    assert len(nodes) == 2
    assert nodes[0].node.id_ == "test_text_node"
    assert nodes[1].node.id_ == "hidden_node"


def _build_zero_score_composable_retriever():
    target = TextNode(text="reachable child", id_="child")
    index = VectorStoreIndex(
        nodes=[],
        objects=[IndexNode(text="sub-index", index_id="sub", obj=target)],
        embed_model=OrthogonalEmbedding(),
    )
    return index.as_retriever(similarity_top_k=1)


def test_composable_retrieval_preserves_zero_score() -> None:
    retriever = _build_zero_score_composable_retriever()

    nodes = retriever.retrieve("orthogonal query")

    assert [(node.node.node_id, node.score) for node in nodes] == [("child", 0.0)]


@pytest.mark.asyncio
async def test_async_composable_retrieval_preserves_zero_score() -> None:
    retriever = _build_zero_score_composable_retriever()

    nodes = await retriever.aretrieve("orthogonal query")

    assert [(node.node.node_id, node.score) for node in nodes] == [("child", 0.0)]


def test_recursive_retriever_preserves_zero_score() -> None:
    retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict={"root": ZeroScoreIndexNodeRetriever()},
        node_dict={"child": TextNode(text="reachable child", id_="child-node")},
    )

    nodes = retriever.retrieve("query")

    assert [(node.node.node_id, node.score) for node in nodes] == [("child-node", 0.0)]


def _build_retriever_with_query_engine_object():
    embed = MockEmbedding(embed_dim=3)
    sub_qe = VectorStoreIndex.from_documents(
        [
            Document(
                text="Paris is the capital of France.",
                metadata={"source": "geography.pdf"},
            )
        ],
        embed_model=embed,
    ).as_query_engine(llm=MockLLM())
    top_index = VectorStoreIndex(
        nodes=[],
        objects=[IndexNode(text="France sub-index", index_id="france-sub", obj=sub_qe)],
        embed_model=embed,
    )
    return top_index.as_retriever(similarity_top_k=1)


def test_query_engine_object_metadata_preserved_sync() -> None:
    retriever = _build_retriever_with_query_engine_object()
    nodes = retriever.retrieve("Capital of France?")
    assert nodes[0].node.metadata


@pytest.mark.asyncio
async def test_query_engine_object_metadata_preserved_async() -> None:
    retriever = _build_retriever_with_query_engine_object()
    nodes = await retriever.aretrieve("Capital of France?")
    assert nodes[0].node.metadata


@pytest.mark.asyncio
async def test_dedup_preserves_nodes_with_different_node_ids() -> None:
    node1 = TextNode(text="shared content", metadata={}, id_="node-1")
    node2 = TextNode(text="shared content", metadata={}, id_="node-2")

    retriever = SummaryIndex(nodes=[node1, node2]).as_retriever()

    sync_nodes = retriever.retrieve("test")
    assert len(sync_nodes) == 2
    assert {n.node.node_id for n in sync_nodes} == {"node-1", "node-2"}

    async_nodes = await retriever.aretrieve("test")
    assert len(async_nodes) == 2
    assert {n.node.node_id for n in async_nodes} == {"node-1", "node-2"}
