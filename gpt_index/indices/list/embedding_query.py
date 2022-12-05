"""Embedding query for list index."""
from typing import List, Optional

from gpt_index.embeddings.utils import (
    cosine_similarity,
    get_query_embedding,
    get_text_embedding,
)
from gpt_index.indices.data_structs import IndexList, Node
from gpt_index.indices.list.query import BaseGPTListIndexQuery
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)


class GPTListIndexEmbeddingQuery(BaseGPTListIndexQuery):
    """GPTListIndex query."""

    def __init__(
        self,
        index_struct: IndexList,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        refine_template: Prompt = DEFAULT_REFINE_PROMPT,
        keyword: Optional[str] = None,
        similarity_top_k: Optional[int] = 3,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            keyword=keyword,
        )
        self.similarity_top_k = similarity_top_k

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        if self.keyword is not None:
            nodes = [node for node in nodes if self.keyword in node.text]

        # top k nodes
        similarities = self._get_query_text_embedding_similarities(query_str, nodes)
        sorted_node_tups = sorted(
            zip(similarities, nodes), key=lambda x: x[0], reverse=True
        )
        sorted_nodes = [n for _, n in sorted_node_tups]
        similarity_top_k = self.similarity_top_k or len(nodes)
        top_k_nodes = sorted_nodes[:similarity_top_k]
        return top_k_nodes

    def _get_query_text_embedding_similarities(
        self, query_str: str, nodes: List[Node]
    ) -> List[float]:
        """Get top nodes by similarity to the query."""
        query_embedding = get_query_embedding(query_str)
        # node_similarities: List[Tuple[float, Node]] = []
        similarities = []
        for node in self.index_struct.nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = get_text_embedding(node.text)
                node.embedding = text_embedding

            similarity = cosine_similarity(query_embedding, text_embedding)
            similarities.append(similarity)

        return similarities
