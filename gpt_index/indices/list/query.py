"""Default query for GPTListIndex."""
from abc import abstractmethod
from typing import Any, List, Optional

from gpt_index.indices.data_structs import IndexList, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)


class BaseGPTListIndexQuery(BaseGPTIndexQuery[IndexList]):
    """GPTListIndex query."""

    def __init__(
        self,
        index_struct: IndexList,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        refine_template: Prompt = DEFAULT_REFINE_PROMPT,
        keyword: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
        self.keyword = keyword

    def _give_response_for_nodes(
        self, query_str: str, nodes: List[Node], verbose: bool = False
    ) -> str:
        """Give response for nodes."""
        response = None
        for node in nodes:
            response = self._query_node(
                query_str,
                node,
                self.text_qa_template,
                self.refine_template,
                response=response,
                verbose=verbose,
            )
        return response or ""

    @abstractmethod
    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        if self.keyword is not None:
            nodes = [node for node in nodes if self.keyword in node.get_text()]
        return nodes

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        nodes = self._get_nodes_for_response(query_str, verbose=verbose)
        return self._give_response_for_nodes(query_str, nodes, verbose=verbose)


class GPTListIndexQuery(BaseGPTListIndexQuery):
    """GPTListIndex query."""

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        if self.keyword is not None:
            nodes = [node for node in nodes if self.keyword in node.get_text()]
        return nodes
