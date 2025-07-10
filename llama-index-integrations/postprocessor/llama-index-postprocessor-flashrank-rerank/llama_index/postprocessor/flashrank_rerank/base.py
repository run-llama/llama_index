from typing import Any
from typing_extensions import override

from flashrank import Ranker, RerankRequest
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


class FlashRankRerank(BaseNodePostprocessor):
    model: str = Field(
        description="FlashRank model name.", default="ms-marco-TinyBERT-L-2-v2"
    )
    top_n: int = Field(
        description="Number of nodes to return sorted by score.", default=20
    )
    max_length: int = Field(
        description="Maximum length of passage text passed to the reranker.",
        default=512,
    )

    _reranker: Ranker = PrivateAttr()

    @override
    def model_post_init(self, context: Any, /) -> None:  # pyright: ignore[reportAny]
        self._reranker = Ranker(model_name=self.model, max_length=self.max_length)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "FlashRankRerank"

    @dispatcher.span
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes: RerankRequest = RerankRequest(
            query=query_bundle.query_str,
            passages=[
                {
                    "id": node.node.id_,
                    "text": node.node.get_content(metadata_mode=MetadataMode.EMBED),
                }
                for node in nodes
            ],
        )
        ## you would need to define a custom event subclassing BaseEvent from llama_index_instrumentation
        dispatcher.event(FlashRerankingQueryEvent(nodes=nodes, model_name=model_name, query_str = query_bundle.query_str, top_k = self.top_n))
        scores = self._reranker.rerank(query_and_nodes)
        scores_by_id = {score["id"]: score["score"] for score in scores}

        if len(scores) != len(nodes):
            msg = "Number of scores and nodes do not match."
            raise ValueError(msg)

         for node in nodes:
            node.score = scores_by_id[node.node.id_]

         new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
         dispatch.event(FlashRerankEndEvent(nodes=new_nodes))

        return new_nodes
