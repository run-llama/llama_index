"""Query for GPTKGTableIndex."""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from gpt_index.data_structs.data_structs import KG, Node
from gpt_index.indices.keyword_table.utils import extract_keywords_given_response
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.utils import truncate_text
from gpt_index.prompts.default_prompts import DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
from gpt_index.prompts.prompts import QueryKeywordExtractPrompt

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class GPTKGTableQuery(BaseGPTIndexQuery[KG]):
    """Base GPT KG Table Index Query.

    Arguments are shared among subclasses.

    Args:
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
        query_keyword_extract_template: Optional[QueryKeywordExtractPrompt] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        include_text: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET
        self._include_text = include_text

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response, _ = self._llm_predictor.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logging.info(f"> Starting query: {query_bundle.query_str}")
        keywords = self._get_keywords(query_bundle.query_str)
        logging.info(f"> Query keywords: {keywords}")
        rel_texts = []
        cur_rel_map = {}
        chunk_indices_count: Dict[str, int] = defaultdict(int)
        for keyword in keywords:
            cur_rel_texts = self.index_struct.get_rel_map_texts(keyword)
            rel_texts.extend(cur_rel_texts)
            cur_rel_map[keyword] = self.index_struct.get_rel_map_tuples(keyword)
            if self._include_text:
                for node_id in self.index_struct.get_node_ids(keyword):
                    chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            list(chunk_indices_count.keys()),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self.num_chunks_per_query]
        sorted_nodes = [
            self.index_struct.text_chunks[idx] for idx in sorted_chunk_indices
        ]
        # filter sorted nodes
        sorted_nodes = [node for node in sorted_nodes if self._should_use_node(node)]
        for chunk_idx, node in zip(sorted_chunk_indices, sorted_nodes):
            logging.info(
                f"> Querying with idx: {chunk_idx}: "
                f"{truncate_text(node.get_text(), 80)}"
            )

        # add relationships as Node
        # TODO: make initial text customizable
        rel_initial_text = (
            "The following are knowledge triplets "
            "in the form of (subset, predicate, object):"
        )
        rel_info = [rel_initial_text] + rel_texts
        rel_node_info = {
            "kg_rel_texts": rel_texts,
            "kg_rel_map": cur_rel_map,
        }
        rel_text_node = Node(text="\n".join(rel_info), node_info=rel_node_info)
        rel_info_text = "\n".join(rel_info)
        logging.info(f"> Extracted relationships: {rel_info_text}")
        sorted_nodes.append(rel_text_node)

        return sorted_nodes

    def _get_extra_info_for_response(
        self, nodes: List[Node]
    ) -> Optional[Dict[str, Any]]:
        """Get extra info for response."""
        for node in nodes:
            if node.node_info is None or "kg_rel_map" not in node.node_info:
                continue
            return node.node_info
        raise ValueError("kg_rel_map must be found in at least one Node.")
