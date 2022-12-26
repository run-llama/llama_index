"""Summarize query."""


from typing import Any, Optional, cast

from gpt_index.data_structs import IndexGraph
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response.builder import ResponseBuilder, ResponseMode, TextChunk
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt


class GPTTreeIndexSummarizeQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index summarize query.

    This class builds a query-specific tree from leaf nodes to return a response.
    Using this query mode means that the tree index doesn't need to be built
    when initialized, since we rebuild the tree for each query.

    .. code-block:: python

        response = index.query("<query_str>", mode="summarize")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        num_children: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct, **kwargs)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT
        self.num_children = num_children

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        index_struct = cast(IndexGraph, self._index_struct)
        sorted_node_list = get_sorted_node_list(index_struct.all_nodes)
        sorted_node_txts = [TextChunk(n.get_text()) for n in sorted_node_list]

        response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            texts=sorted_node_txts,
        )
        response = response_builder.get_response(
            query_str,
            verbose=verbose,
            mode=ResponseMode.TREE_SUMMARIZE,
            num_children=self.num_children,
        )
        return response
