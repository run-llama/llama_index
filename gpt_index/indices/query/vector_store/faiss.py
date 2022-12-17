"""Default query for GPTFaissIndex."""
from abc import abstractmethod
from typing import Any, List, Optional
import numpy as np

from gpt_index.indices.data_structs import IndexDict, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.embeddings.openai import OpenAIEmbedding


class GPTFaissIndexQuery(BaseGPTIndexQuery[IndexDict]):
    """GPTFaissIndex query."""

    def __init__(
        self,
        index_struct: IndexDict,
        text_qa_template: Optional[Prompt] = None,
        refine_template: Optional[Prompt] = None,
        faiss_index: Optional[Any] = None,
        embed_model: Optional[OpenAIEmbedding] = None,
        similarity_top_k: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT
        self._faiss_index = faiss_index
        self._embed_model = embed_model or OpenAIEmbedding()
        self.similarity_top_k = similarity_top_k

    def _give_response_for_nodes(
        self, query_str: str, nodes: List[Node], verbose: bool = False
    ) -> str:
        """Give response for nodes."""
        response = None
        for node in nodes:
            response = self._query_node(
                query_str,
                node,
                self.text_qa_template,
                self.refine_template,
                response=response,
                verbose=verbose,
            )
        return response or ""

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        query_embedding_np = np.array(query_embedding)[np.newaxis, :]


    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        nodes = self._get_nodes_for_response(query_str, verbose=verbose)
        return self._give_response_for_nodes(query_str, nodes, verbose=verbose)