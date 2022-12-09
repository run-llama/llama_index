"""Embedding query for list index."""
from typing import List, Optional

from gpt_index.embeddings.openai import OpenAIEmbedding
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
        similarity_top_k: Optional[int] = 1,
        embed_model: Optional[OpenAIEmbedding] = None,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            keyword=keyword,
        )
        self._embed_model = embed_model or OpenAIEmbedding()
        self.similarity_top_k = similarity_top_k

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        if self.keyword is not None:
            nodes = [node for node in nodes if self.keyword in node.get_text()]

        # top k nodes
        similarities = self._get_query_text_embedding_similarities(query_str, nodes)
        sorted_node_tups = sorted(
            zip(similarities, nodes), key=lambda x: x[0], reverse=True
        )
        sorted_nodes = [n for _, n in sorted_node_tups]
        similarity_top_k = self.similarity_top_k or len(nodes)
        top_k_nodes = sorted_nodes[:similarity_top_k]
        if verbose:
            top_k_node_text = "\n".join([n.get_text() for n in top_k_nodes])
            print(f"Top {similarity_top_k} nodes: {top_k_node_text}")
        return top_k_nodes

    def _get_query_text_embedding_similarities(
        self, query_str: str, nodes: List[Node]
    ) -> List[float]:
        """Get top nodes by similarity to the query."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        similarities = []
        for node in self.index_struct.nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = self._embed_model.get_text_embedding(node.get_text())
                node.embedding = text_embedding

            similarity = self._embed_model.similarity(query_embedding, text_embedding)
            similarities.append(similarity)

        return similarities
