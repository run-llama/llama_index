"""Default query for GPTListIndex."""
from abc import abstractmethod
from typing import Any, List, Optional

from gpt_index.indices.data_structs import IndexList, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt


class BaseGPTListIndexQuery(BaseGPTIndexQuery[IndexList]):
    """GPTListIndex query.

    Arguments are shared among subclasses.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        keyword (Optional[str]): If specified, keyword to filter nodes.
            Simulates Ctrl+F lookup in a document.

    """

    def __init__(
        self,
        index_struct: IndexList,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        keyword: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT
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

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        nodes = self._get_nodes_for_response(query_str, verbose=verbose)
        return self._give_response_for_nodes(query_str, nodes, verbose=verbose)


class GPTListIndexQuery(BaseGPTListIndexQuery):
    """GPTListIndex query.

    The default query mode for GPTListIndex, which traverses
    each node in sequence and synthesizes a response across all nodes
    (with an optional keyword filter).
    Set when `mode="default"` in `query` method of `GPTListIndex`.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    See BaseGPTListIndexQuery for arguments.

    """

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        if self.keyword is not None:
            nodes = [node for node in nodes if self.keyword in node.get_text()]
        return nodes
