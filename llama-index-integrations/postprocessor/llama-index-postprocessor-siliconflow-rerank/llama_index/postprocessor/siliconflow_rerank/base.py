import requests
from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/rerank"

dispatcher = get_dispatcher(__name__)

AVAILABLE_OPTIONS = [
    "BAAI/bge-reranker-v2-m3",
    "Bnetease-youdao/bce-reranker-base_v1",
]


class SiliconFlowRerank(BaseNodePostprocessor):
    model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Specifies the model to be used.",
    )
    base_url: str = Field(
        default=DEFAULT_SILICONFLOW_API_URL,
        description="The URL of the SiliconFlow Rerank API.",
    )
    api_key: str = Field(default=None, description="The SiliconFlow API key.")

    top_n: int = Field(
        description="Number of most relevant documents or indices to return."
    )
    return_documents: bool = Field(
        default=True,
        description="Specify whether the response should include the document text.",
    )
    max_chunks_per_doc: int = Field(
        default=1024,
        description="""\
            Maximum number of chunks generated from within a document.
            Long documents are divided into multiple chunks for calculation,
            and the highest score among the chunks is taken as the document's score.
        """,
    )
    overlap_tokens: int = Field(
        default=80,
        description="Number of token overlaps between adjacent chunks when documents are chunked.",
    )

    _session: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = DEFAULT_SILICONFLOW_API_URL,
        api_key: Optional[str] = None,
        top_n: int = 4,
        return_documents: bool = True,
        max_chunks_per_doc: int = 1024,
        overlap_tokens: int = 80,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_n=top_n,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            overlap_tokens=overlap_tokens,
        )
        self._session: requests.Session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @classmethod
    def class_name(cls) -> str:
        return "SiliconFlowRerank"

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "return_documents": self.return_documents,
            "max_chunks_per_doc": self.max_chunks_per_doc,
            "overlap_tokens": self.overlap_tokens,
        }

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
            response = self._session.post(
                self.base_url,
                json={
                    "model": self.model,
                    "query": query_bundle.query_str,
                    "documents": texts,
                    "top_n": self.top_n,
                    **self._model_kwargs,
                },
            ).json()
            if "results" not in response:
                raise RuntimeError(response)

            new_nodes = []
            for result in response["results"]:
                new_node_with_score = NodeWithScore(
                    node=nodes[result["index"]].node, score=result["relevance_score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
