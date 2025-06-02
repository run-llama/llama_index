from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from google.cloud.aiplatform import telemetry
from vertexai.preview import rag


class VertexAIRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_name: str,
        similarity_top_k: Optional[int] = None,
        vector_distance_threshold: Optional[float] = 0.3,
        user_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vertex AI Retriever."""
        self.rag_resources = [rag.RagResource(rag_corpus=corpus_name)]
        self._similarity_top_k = similarity_top_k
        self._vector_distance_threshold = vector_distance_threshold
        self._user_agent = user_agent or "llama-index/0.0.0"

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from the platform."""
        with telemetry.tool_context_manager(self._user_agent):
            response = rag.retrieval_query(
                text=query_bundle.query_str,
                rag_resources=self.rag_resources,
                similarity_top_k=self._similarity_top_k,
                vector_distance_threshold=self._vector_distance_threshold,
            )

        if response.contexts:
            return [
                NodeWithScore(
                    node=TextNode(
                        text=context.text,
                        metadata={
                            "source_uri": context.source_uri,
                            "source_display_name": context.source_display_name,
                        },
                    ),
                    score=context.distance,
                )
                for context in response.contexts.contexts
            ]
        else:
            return []

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        return self._retrieve(query_bundle=query_bundle)
