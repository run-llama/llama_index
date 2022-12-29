"""Retrieve query."""

from gpt_index.data_structs.data_structs import IndexGraph
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response.builder import ResponseBuilder, TextChunk
from gpt_index.indices.utils import get_sorted_node_list


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

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        node_list = get_sorted_node_list(self.index_struct.root_nodes)
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        node_text = self._prompt_helper.get_text_from_nodes(
            node_list, prompt=text_qa_template
        )
        response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            texts=[TextChunk(node_text)],
        )
        response = response_builder.get_response(
            query_str, verbose=verbose, mode=self._response_mode
        )
        return response
