"""Summarize query."""


from typing import Any, Optional

from gpt_index.indices.common.tree.base import GPTTreeIndexBuilder
from gpt_index.indices.data_structs import IndexGraph
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response_utils.response import give_response
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt


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
        num_children: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct, **kwargs)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.num_children = num_children

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")

        # use prompt composability to build a summary prompt
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(text_qa_template)

        index_builder = GPTTreeIndexBuilder(
            self.num_children,
            summary_template,
            self._llm_predictor,
            self._prompt_helper,
        )
        all_nodes = self._index_struct.all_nodes.copy()
        root_nodes = index_builder.build_index_from_nodes(
            all_nodes, all_nodes, verbose=verbose
        )

        node_list = get_sorted_node_list(root_nodes)
        node_text = self._prompt_helper.get_text_from_nodes(
            node_list, prompt=self.text_qa_template
        )
        response = give_response(
            self._prompt_helper,
            self._llm_predictor,
            query_str,
            node_text,
            text_qa_template=self.text_qa_template,
            verbose=verbose,
        )
        return response
