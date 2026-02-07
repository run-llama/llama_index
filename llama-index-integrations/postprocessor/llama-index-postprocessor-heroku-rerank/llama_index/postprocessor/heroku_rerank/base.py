"""Heroku Inference API reranking postprocessor for LlamaIndex."""

from typing import Any, List, Optional

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class HerokuRerank(BaseNodePostprocessor):
    """Heroku Inference API reranking postprocessor.

    This class wraps the Heroku Inference API's /v1/rerank endpoint,
    which provides access to Cohere reranking models.

    Args:
        api_key: Your Heroku Inference API key.
        model: The reranking model to use (default: "cohere-rerank-3-5").
        base_url: The Heroku Inference API base URL.
        top_n: Number of top documents to return after reranking.
        timeout: Request timeout in seconds.

    Example:
        >>> from llama_index.postprocessor.heroku_rerank import HerokuRerank
        >>> reranker = HerokuRerank(api_key="your-key", top_n=5)
        >>> query_engine = index.as_query_engine(node_postprocessors=[reranker])
    """

    api_key: str = Field(description="Heroku Inference API key")
    base_url: str = Field(
        default="https://us.inference.heroku.com",
        description="Heroku Inference API base URL",
    )
    model: str = Field(
        default="cohere-rerank-3-5",
        description="The reranking model to use",
    )
    top_n: int = Field(
        default=5,
        description="Number of top documents to return after reranking",
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
    )

    _client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "cohere-rerank-3-5",
        base_url: str = "https://us.inference.heroku.com",
        top_n: int = 5,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Heroku reranking postprocessor."""
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            top_n=top_n,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes based on relevance to the query.

        Args:
            nodes: List of nodes with scores to rerank.
            query_bundle: The query to rerank against.

        Returns:
            Reranked list of nodes, limited to top_n.
        """
        if not query_bundle or not nodes:
            return nodes

        # Extract document content from nodes
        documents = [node.node.get_content() for node in nodes]

        # Call the reranking API
        response = self._client.post(
            f"{self.base_url}/v1/rerank",
            headers=self._get_headers(),
            json={
                "model": self.model,
                "query": query_bundle.query_str,
                "documents": documents,
                "top_n": min(self.top_n, len(nodes)),
            },
        )
        response.raise_for_status()

        # Process results and reorder nodes
        results = response.json()["results"]
        reranked_nodes = []

        for result in results:
            idx = result["index"]
            score = result["relevance_score"]
            # Create a new NodeWithScore with the rerank score
            node_with_score = NodeWithScore(
                node=nodes[idx].node,
                score=score,
            )
            reranked_nodes.append(node_with_score)

        return reranked_nodes

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Async rerank nodes based on relevance to the query.

        Args:
            nodes: List of nodes with scores to rerank.
            query_bundle: The query to rerank against.

        Returns:
            Reranked list of nodes, limited to top_n.
        """
        if not query_bundle or not nodes:
            return nodes

        # Extract document content from nodes
        documents = [node.node.get_content() for node in nodes]

        # Call the reranking API
        response = await self._async_client.post(
            f"{self.base_url}/v1/rerank",
            headers=self._get_headers(),
            json={
                "model": self.model,
                "query": query_bundle.query_str,
                "documents": documents,
                "top_n": min(self.top_n, len(nodes)),
            },
        )
        response.raise_for_status()

        # Process results and reorder nodes
        results = response.json()["results"]
        reranked_nodes = []

        for result in results:
            idx = result["index"]
            score = result["relevance_score"]
            # Create a new NodeWithScore with the rerank score
            node_with_score = NodeWithScore(
                node=nodes[idx].node,
                score=score,
            )
            reranked_nodes.append(node_with_score)

        return reranked_nodes

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "HerokuRerank"
