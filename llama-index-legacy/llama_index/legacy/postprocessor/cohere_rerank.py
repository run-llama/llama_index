import os
from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CBEventType, EventPayload
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
from llama_index.legacy.schema import NodeWithScore, QueryBundle


class CohereRerank(BaseNodePostprocessor):
    model: str = Field(description="Cohere model name.")
    top_n: int = Field(description="Top N nodes to return.")

    _client: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "rerank-english-v2.0",
        api_key: Optional[str] = None,
    ):
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        try:
            from cohere import Client
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        self._client = Client(api_key=api_key)
        super().__init__(top_n=top_n, model=model)

    @classmethod
    def class_name(cls) -> str:
        return "CohereRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
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
            texts = [node.node.get_content() for node in nodes]
            results = self._client.rerank(
                model=self.model,
                top_n=self.top_n,
                query=query_bundle.query_str,
                documents=texts,
            )

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
