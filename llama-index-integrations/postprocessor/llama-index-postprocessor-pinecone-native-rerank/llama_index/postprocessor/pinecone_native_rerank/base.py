import os
from typing import List, Optional, Literal
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)


# as of today 2024/12/7
# pinecone itself supports 3 rerank models out of its sdk
# https://app.pinecone.io/organizations/-Nn577_974iRsvC6nVxg/projects/a4fe57a4-b1cc-4a99-bf1d-c35a595cae4a/models
class PineconeNativeRerank(BaseNodePostprocessor):
    model: Literal["bge-reranker-v2-m3", "cohere-rerank-3.5", "pinecone-rerank-v0"] = (
        Field(
            description="supported Pinecone inference rerank model name",
            default="bge-reranker-v2-m3",
        )
    )
    top_n: int = Field(description="Top N nodes to return")

    _pc: any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "bge-reranker-v2-m3",
        api_key: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        try:
            api_key = api_key or os.environ["PINECONE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in pinecone api key or "
                "specify via PINECONE_API_KEY environment variable "
            )

        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "Cannot import pinecone package, please `pip install pinecone-client`."
            )

        self._pc = Pinecone(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "PineconeNativeRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model
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

            reranked_result = self._pc.inference.rerank(
                model=self.model,
                top_n=self.top_n,
                query=query_bundle.query_str,
                documents=texts,
                return_documents=True,
            )
            new_nodes = []
            for result in reranked_result.data:
                new_nodes.append(
                    NodeWithScore(
                        node=nodes[result.index].node,
                        score=result.score,
                    )
                )
            event.on_end(payload={EventPayload.NODES: new_nodes})
        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
