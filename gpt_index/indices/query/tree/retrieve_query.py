"""Retrieve query."""
import logging
from typing import List, Optional

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
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

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logging.info(f"> Starting query: {query_bundle.query_str}")
        node_list = get_sorted_node_list(self.index_struct.root_nodes)
        text_qa_template = self.text_qa_template.partial_format(
            query_str=query_bundle.query_str
        )
        node_text = self._prompt_helper.get_text_from_nodes(
            node_list, prompt=text_qa_template
        )
        return [Node(text=node_text)]
