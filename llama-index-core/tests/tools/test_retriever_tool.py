"""Test retriever tool."""
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.tools import RetrieverTool
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class MockRetriever(BaseRetriever):
    """Custom retriever for testing."""

    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [NodeWithScore(node=TextNode(text=f"mock_{query_str}"), score=0.9)]

    async def _aretrieve(self, query_str: str) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [NodeWithScore(node=TextNode(text=f"mock_{query_str}"), score=0.9)]


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
    assert str(response_nodes) == "mock_hello world\n\n\n\n"
    assert response_nodes.raw_output[0].node.text == "mock_hello world\n\n"

    # Test node_postprocessors
    node_postprocessors = [MockPostProcessor()]
    pr_retriever_tool = RetrieverTool.from_defaults(
        retriever=retriever, node_postprocessors=node_postprocessors
    )
    pr_response_nodes = pr_retriever_tool("hello world")
    assert str(pr_response_nodes) == "processed_mock_hello world\n\n\n\n"
