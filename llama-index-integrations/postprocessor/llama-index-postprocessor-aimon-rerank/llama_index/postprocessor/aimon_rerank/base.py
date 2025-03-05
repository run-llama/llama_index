import os
from typing import Any, List, Optional

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


class AIMonRerank(BaseNodePostprocessor):

    model: str = Field(description="AIMon's reranking model name.")
    top_n: int = Field(description="Top N nodes to return.")
    task_definition: str = Field(
        default="Determine the relevance of context documents with respect to the user query.",
        description="The task definition for the AIMon reranker.",
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "rr",
        api_key: Optional[str] = None,
        task_definition: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        self.task_definition = task_definition or (
            "Determine the relevance of context documents with respect to the user query."
        )
        try:
            api_key = api_key or os.environ["AIMON_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in AIMon api key or specify via AIMON_API_KEY environment variable"
            )
        try:
            from aimon import Client
        except ImportError:
            raise ImportError(
                "Cannot import AIMon package, please `pip install aimon`."
            )

        self._client = Client(auth_header="Bearer {}".format(api_key))

    @classmethod
    def class_name(cls) -> str:
        return "AIMonRerank"

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
        if not nodes:
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
            # Get the text content from each node.
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
            # Retrieve scores; AIMon returns a List[List[float]].
            scores = self._client.retrieval.rerank(
                context_docs=texts,
                queries=[query_bundle.query_str],  # Single query wrapped in a list.
                task_definition=self.task_definition,
            )
            scores_list = scores[0]

            # Pair each score with its original document index.
            indexed_scores = list(enumerate(scores_list))
            # Sort by score in descending order.
            sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
            # Take only the top_n documents.
            if self.top_n is not None:
                sorted_scores = sorted_scores[: self.top_n]

            new_nodes = []
            for idx, score in sorted_scores:
                new_node_with_score = NodeWithScore(
                    node=nodes[idx].node,
                    score=score,
                )
                # Explicitly update the similarity field if needed.
                new_node_with_score.similarity = score
                new_nodes.append(new_node_with_score)

            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
