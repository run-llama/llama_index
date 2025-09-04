"""Test retriever tool."""

from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.tools import RetrieverTool
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import pytest


class MockRetriever(BaseRetriever):
    """Custom retriever for testing."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [
            NodeWithScore(
                node=TextNode(
                    text=f"mock_{query_bundle}",
                    text_template="Metadata:\n{metadata_str}\n\nContent:\n{content}",
                    metadata_template="- {key}: {value}",
                    metadata={"key": "value"},
                ),
                score=0.9,
            )
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [
            NodeWithScore(
                node=TextNode(
                    text=f"mock_{query_bundle}",
                    text_template="Metadata:\n{metadata_str}\n\nContent:\n{content}",
                    metadata_template="- {key}: {value}",
                    metadata={"key": "value"},
                ),
                score=0.9,
            )
        ]


class MockPostProcessor(BaseNodePostprocessor):
    @classmethod
    def class_name(cls) -> str:
        return "CitationPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for n in nodes:
            prefix = f"processed_"
            n.node.text = prefix + n.node.text
        return nodes


def test_retriever_tool() -> None:
    """Test retriever tool."""
    # Test retrieval
    retriever = MockRetriever()
    retriever_tool = RetrieverTool.from_defaults(retriever=retriever)
    response_nodes = retriever_tool("hello world")
    assert (
        str(response_nodes)
        == "Metadata:\n- key: value\n\nContent:\nmock_hello world\n\n"
    )
    assert response_nodes.raw_output[0].node.text == "mock_hello world\n\n"

    # Test node_postprocessors
    node_postprocessors = [MockPostProcessor()]
    pr_retriever_tool = RetrieverTool.from_defaults(
        retriever=retriever, node_postprocessors=node_postprocessors
    )
    pr_response_nodes = pr_retriever_tool("hello world")
    assert (
        str(pr_response_nodes)
        == "Metadata:\n- key: value\n\nContent:\nprocessed_mock_hello world\n\n"
    )


@pytest.mark.asyncio
async def test_retriever_tool_async() -> None:
    """Test retriever tool async call."""
    # Test async retrieval
    retriever = MockRetriever()
    retriever_tool = RetrieverTool.from_defaults(retriever=retriever)
    response_nodes = await retriever_tool.acall("hello world")
    assert (
        str(response_nodes)
        == "Metadata:\n- key: value\n\nContent:\nmock_hello world\n\n"
    )
    assert response_nodes.raw_output[0].node.text == "mock_hello world\n\n"

    # Test node_postprocessors async
    node_postprocessors = [MockPostProcessor()]
    pr_retriever_tool = RetrieverTool.from_defaults(
        retriever=retriever, node_postprocessors=node_postprocessors
    )
    pr_response_nodes = await pr_retriever_tool.acall("hello world")
    assert (
        str(pr_response_nodes)
        == "Metadata:\n- key: value\n\nContent:\nprocessed_mock_hello world\n\n"
    )
