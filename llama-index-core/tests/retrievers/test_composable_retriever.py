import pytest

from llama_index.core import MockEmbedding, VectorStoreIndex
from llama_index.core.indices import SummaryIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import (
    Document,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


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
async def test_dedup_preserves_nodes_from_different_docs() -> None:
    node1 = TextNode(text="shared content", metadata={}, id_="node-1")
    node1.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-1")
    node2 = TextNode(text="shared content", metadata={}, id_="node-2")
    node2.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-2")

    retriever = SummaryIndex(nodes=[node1, node2]).as_retriever()

    sync_nodes = retriever.retrieve("test")
    assert len(sync_nodes) == 2
    assert {n.node.ref_doc_id for n in sync_nodes} == {"doc-1", "doc-2"}

    async_nodes = await retriever.aretrieve("test")
    assert len(async_nodes) == 2
    assert {n.node.ref_doc_id for n in async_nodes} == {"doc-1", "doc-2"}
