import logging
import os
from typing import Any, Callable, List, Optional

import cohere
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)

DEFAULT_MAX_RETRIES = 10


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    """Create a retry decorator for Cohere API calls."""
    min_seconds = 4
    max_seconds = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(
                (
                    cohere.errors.ServiceUnavailableError,
                    cohere.errors.InternalServerError,
                    cohere.errors.GatewayTimeoutError,
                )
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class CohereRerank(BaseNodePostprocessor):
    model: str = Field(description="Cohere model name.")
    top_n: int = Field(description="Top N nodes to return.")
    base_url: Optional[str] = Field(description="Cohere base url.", default=None)
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Maximum number of retries for API calls.",
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        super().__init__(top_n=top_n, model=model, max_retries=max_retries)
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        try:
            from cohere import ClientV2
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        self._client = ClientV2(api_key=api_key, base_url=base_url)

    @classmethod
    def class_name(cls) -> str:
        return "CohereRerank"

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

            retry_decorator = _create_retry_decorator(max_retries=self.max_retries)

            @retry_decorator
            def rerank_with_retry() -> Any:
                return self._client.rerank(
                    model=self.model,
                    top_n=self.top_n,
                    query=query_bundle.query_str,
                    documents=texts,
                )

            results = rerank_with_retry()

            new_nodes = []
            for result in results.results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
