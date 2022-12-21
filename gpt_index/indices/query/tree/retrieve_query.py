"""Retrieve query."""

from typing import Any, Optional

from gpt_index.indices.data_structs import IndexGraph
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response_utils.response import give_response
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt


class GPTTreeIndexRetQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index retrieve query.

    This class directly retrieves the answer from the root nodes.

    Unlike GPTTreeIndexLeafQuery, this class assumes the graph already stores
    the answer (because it was constructed with a query_str), so it does not
    attempt to parse information down the graph in order to synthesize an answer.

    .. code-block:: python

        response = index.query("<query_str>", mode="retrieve")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct, **kwargs)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        node_list = get_sorted_node_list(self.index_struct.root_nodes)
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
