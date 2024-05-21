from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from vertexai.preview import rag


class VertexAIRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_name: str,
        similarity_top_k: Optional[int] = None,
        vector_distance_threshold: Optional[float] = 0.3,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vertex AI Retriever."""
        self.rag_resources = [rag.RagResource(rag_corpus=corpus_name)]
        self._similarity_top_k = similarity_top_k
        self._vector_distance_threshold = vector_distance_threshold

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from the platform."""
        response = rag.retrieval_query(
            text=query_bundle.query_str,
            rag_resources=self.rag_resources,
            similarity_top_k=self._similarity_top_k,
            vector_distance_threshold=self._vector_distance_threshold,
        )

        return [
            NodeWithScore(node=TextNode(text=context.text), score=context.distance)
            for context in response.rag_contexts.contexts
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        return self._retrieve(query_bundle=query_bundle)
