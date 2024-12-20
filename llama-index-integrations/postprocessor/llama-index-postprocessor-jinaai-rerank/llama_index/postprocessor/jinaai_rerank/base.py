from typing import Any, List, Optional
import requests

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

DEFAULT_JINA_AI_API_URL = "https://api.jina.ai/v1"

dispatcher = get_dispatcher(__name__)


class JinaRerank(BaseNodePostprocessor):
    api_url: str = Field(
        default=f"{DEFAULT_JINA_AI_API_URL}/rerank",
        description="The URL of the JinaAI Rerank API.",
    )
    api_key: str = Field(default=None, description="The JinaAI API key.")
    model: str = Field(
        default="jina-reranker-v1-base-en",
        description="The model to use when calling Jina AI API",
    )

    top_n: int = Field(description="Top N nodes to return.")

    _session: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "jina-reranker-v1-base-en",
        base_url: str = DEFAULT_JINA_AI_API_URL,
        api_key: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        self.api_url = f"{base_url}/rerank"
        self.api_key = get_from_param_or_env("api_key", api_key, "JINAAI_API_KEY", "")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"}
        )

    @classmethod
    def class_name(cls) -> str:
        return "JinaRerank"

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
                model_name=self.model,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
            resp = self._session.post(  # type: ignore
                self.api_url,
                json={
                    "query": query_bundle.query_str,
                    "documents": texts,
                    "model": self.model,
                    "top_n": self.top_n,
                },
            ).json()
            if "results" not in resp:
                raise RuntimeError(resp["detail"])

            results = resp["results"]

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result["index"]].node, score=result["relevance_score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
