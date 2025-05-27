from typing import Callable, List, Optional, Union

from llama_index.core.bridge.pydantic import Field
from typing import List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
import httpx
import warnings

DEFAULT_URL = "http://127.0.0.1:8080"
TOP_N = 5

dispatcher = get_dispatcher(__name__)


class TextEmbeddingInference(BaseNodePostprocessor):
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    top_n: int = Field(
        default=TOP_N, description="Number of nodes to return sorted by score."
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    truncate_text: bool = Field(
        default=True,
        description="Whether to truncate text or not when generating embeddings.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )
    model_name: str = Field(
        default="API",
        description="Base URL for the text embeddings service.",
    )

    mode: str = Field(
        default="text",
        description="Re-ranking Method, full for including meta-data too.",
    )

    def __init__(
        self,
        top_n: int = TOP_N,
        base_url: str = DEFAULT_URL,
        text_instruction: Optional[str] = None,
        query_instruction: Optional[str] = None,
        timeout: float = 60.0,
        truncate_text: bool = True,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
        model_name="API",
    ):
        super().__init__(
            base_url=base_url,
            top_n=top_n,
            text_instruction=text_instruction,
            query_instruction=query_instruction,
            timeout=timeout,
            truncate_text=truncate_text,
            auth_token=auth_token,
            model_name=model_name,
            mode="text",
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingsInference"

    def _call_api(self, query: str, texts: List[str]) -> List[float]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token

        json_data = {"query": query, "texts": texts}

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/rerank",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_name,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query = query_bundle.query_str
        if self.mode == "full":
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
        elif self.mode == "text":
            texts = [node.text for node in nodes]
        else:
            warnings.warn('Re-Ranking Mode defaulting to mode "text"')
            texts = [node.text for node in nodes]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._call_api(query, texts)
            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = float(score["score"])

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
