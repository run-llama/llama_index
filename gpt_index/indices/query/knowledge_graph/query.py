"""Query for GPTKGTableIndex."""
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from gpt_index.data_structs.data_structs import KG, Node
from gpt_index.indices.keyword_table.utils import (
    extract_keywords_given_response,
    rake_extract_keywords,
    simple_extract_keywords,
)
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.utils import truncate_text
from gpt_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from gpt_index.prompts.prompts import KnowledgeGraphPrompt, QueryKeywordExtractPrompt

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class GPTKGTableQuery(BaseGPTIndexQuery[KG]):
    """Base GPT KG Table Index Query.

    Arguments are shared among subclasses.

    Args:
        keyword_extract_template (Optional[KGExtractPrompt]): A KG
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        query_keyword_extract_template (Optional[QueryKGExtractPrompt]): A Query
            KG Extraction
            Prompt (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.

    """

    def __init__(
        self,
        index_struct: KG,
        keyword_extract_template: Optional[KnowledgeGraphPrompt] = None,
        query_keyword_extract_template: Optional[QueryKeywordExtractPrompt] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET

    @abstractmethod
    def _get_keywords(self, query_str: str, verbose: bool = False) -> List[str]:
        """Extract keywords."""

    def _get_nodes_for_response(
        self,
        query_str: str,
        verbose: bool = False,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        print(f"> Starting query: {query_str}")
        # keywords = self._get_keywords(query_str, verbose=verbose)
        keywords = ["Terry Winograd"]
        print(f"query keywords: {keywords}")
        nodes: List[Node] = []
        for keyword in keywords:
            for chunk in self.index_struct.get_texts(keyword):
                nodes.append(chunk)
        return nodes
