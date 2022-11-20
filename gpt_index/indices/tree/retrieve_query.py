"""Retrieve query."""

from gpt_index.indices.base import BaseGPTIndexQuery
from gpt_index.indices.data_structs import IndexGraph
from gpt_index.indices.response_utils.response import give_response
from gpt_index.indices.utils import get_sorted_node_list, get_text_from_nodes
from gpt_index.prompts.base import Prompt, validate_prompt
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT


class GPTTreeIndexRetQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index retrieve query.

    This class directly retrieves the answer from the root nodes.

    Unlike GPTTreeIndexLeafQuery, this class assumes the graph already stores
    the answer (because it was constructed with a query_str), so it does not
    attempt to parse information down the graph in order to synthesize an answer.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct)
        self.text_qa_template = text_qa_template
        validate_prompt(self.text_qa_template, ["context_str", "query_str"])

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        node_list = get_sorted_node_list(self.index_struct.root_nodes)
        node_text = get_text_from_nodes(node_list)
        response = give_response(
            query_str,
            node_text,
            text_qa_template=self.text_qa_template,
            verbose=verbose,
        )
        return response
