"""GPU-Bridge reranker for LlamaIndex."""

from typing import Any, Dict, List, Optional

import requests
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

GPUBRIDGE_API_URL = "https://api.gpubridge.io/run"


class GPUBridgeRerank(BaseNodePostprocessor):
    """GPU-Bridge semantic reranker.

    Reranks retrieved nodes using GPU-Bridge's reranking service.
    Significantly improves RAG quality by reordering candidates by relevance.

    Install: ``pip install llama-index-postprocessor-gpubridge-rerank``

    .. code-block:: python

        from llama_index.postprocessor.gpubridge_rerank import GPUBridgeRerank

        reranker = GPUBridgeRerank(api_key="gpub_...", top_n=3)

    """

    api_key: Optional[str] = Field(
        default=None,
        description="GPU-Bridge API key. Register at https://gpubridge.io",
    )
    service: str = Field(
        default="rerank",
        description="GPU-Bridge reranking service.",
    )
    top_n: int = Field(
        default=3,
        description="Number of top results to return after reranking.",
    )
    base_url: str = Field(
        default=GPUBRIDGE_API_URL,
        description="GPU-Bridge API endpoint.",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        top_n: int = 3,
        service: str = "rerank",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            top_n=top_n,
            service=service,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GPUBridgeRerank"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        texts = [n.node.get_content() for n in nodes]

        payload = {
            "service": self.service,
            "input": {
                "query": query,
                "documents": texts,
                "top_n": self.top_n,
            },
        }

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            resp = requests.post(
                self.base_url,
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise ValueError(f"GPU-Bridge error: {data['error']}")

            output = data.get("output", {})
            results = output.get("results", [])

            # results is list of {index, score, document}
            reranked = []
            for r in results[: self.top_n]:
                idx = r.get("index", 0)
                score = r.get("score", 0.0)
                if idx < len(nodes):
                    node = nodes[idx]
                    reranked.append(NodeWithScore(node=node.node, score=score))

            event.on_end(payload={EventPayload.NODES: reranked})

        return reranked
