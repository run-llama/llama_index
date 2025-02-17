"""Retriever tool."""

from typing import TYPE_CHECKING, Any, List, Optional


from llama_index.core.base.base_retriever import BaseRetriever

if TYPE_CHECKING:
    from llama_index.core.langchain_helpers.agents.tools import LlamaIndexTool
from llama_index.core.schema import (
    MetadataMode,
    Node,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.postprocessor.types import BaseNodePostprocessor

DEFAULT_NAME = "retriever_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and retrieving a set of relevant documents.
"""


class RetrieverTool(AsyncBaseTool):
    """Retriever tool.

    A tool making use of a retriever.

    Args:
        retriever (BaseRetriever): A retriever.
        metadata (ToolMetadata): The associated metadata of the query engine.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
            node postprocessors.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        metadata: ToolMetadata,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ) -> None:
        self._retriever = retriever
        self._metadata = metadata
        self._node_postprocessors = node_postprocessors or []

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "RetrieverTool":
        name = name or DEFAULT_NAME
        description = description or DEFAULT_DESCRIPTION

        metadata = ToolMetadata(name=name, description=description)
        return cls(
            retriever=retriever,
            metadata=metadata,
            node_postprocessors=node_postprocessors,
        )

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = ""
        if args is not None:
            query_str += ", ".join([str(arg) for arg in args]) + "\n"
        if kwargs is not None:
            query_str += (
                ", ".join([f"{k!s} is {v!s}" for k, v in kwargs.items()]) + "\n"
            )
        if query_str == "":
            raise ValueError("Cannot call query engine without inputs")

        docs = self._retriever.retrieve(query_str)
        docs = self._apply_node_postprocessors(docs, QueryBundle(query_str))
        content = ""
        for doc in docs:
            assert isinstance(doc.node, (Node, TextNode))
            node_copy = doc.node.model_copy()
            node_copy.text_template = "{metadata_str}\n{content}"
            node_copy.metadata_template = "{key} = {value}"
            content += node_copy.get_content(MetadataMode.LLM) + "\n\n"
        return ToolOutput(
            content=content,
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=docs,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = ""
        if args is not None:
            query_str += ", ".join([str(arg) for arg in args]) + "\n"
        if kwargs is not None:
            query_str += (
                ", ".join([f"{k!s} is {v!s}" for k, v in kwargs.items()]) + "\n"
            )
        if query_str == "":
            raise ValueError("Cannot call query engine without inputs")
        docs = await self._retriever.aretrieve(query_str)
        content = ""
        docs = self._apply_node_postprocessors(docs, QueryBundle(query_str))
        for doc in docs:
            assert isinstance(doc.node, (Node, TextNode))
            node_copy = doc.node.model_copy()
            node_copy.text_template = "{metadata_str}\n{content}"
            node_copy.metadata_template = "{key} = {value}"
            content += node_copy.get_content(MetadataMode.LLM) + "\n\n"
        return ToolOutput(
            content=content,
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=docs,
        )

    def as_langchain_tool(self) -> "LlamaIndexTool":
        raise NotImplementedError("`as_langchain_tool` not implemented here.")

    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes
