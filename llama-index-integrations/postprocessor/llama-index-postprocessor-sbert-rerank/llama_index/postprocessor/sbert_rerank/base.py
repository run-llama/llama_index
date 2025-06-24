from typing import Any, List, Optional, Union
from pathlib import Path

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512

dispatcher = get_dispatcher(__name__)


class SentenceTransformerRerank(BaseNodePostprocessor):
    model: str = Field(description="Sentence transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    cross_encoder_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments for CrossEncoder initialization. "
        "device and model should not be included here.",
    )
    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
        cache_dir: Optional[Union[str, Path]] = None,
        cross_encoder_kwargs: Optional[dict] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers or torch package,",
                "please `pip install torch sentence-transformers`",
            )

        super().__init__(
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
            cross_encoder_kwargs=cross_encoder_kwargs or {},
        )

        init_kwargs = self.cross_encoder_kwargs.copy()
        if "device" in init_kwargs or "model" in init_kwargs:
            raise ValueError(
                "'device' and 'model' should not be specified in 'cross_encoder_kwargs'. "
                "Use the top-level 'device' and 'model' parameters instead."
            )

        # Set default max_length if not provided by the user in kwargs.
        if "max_length" not in init_kwargs:
            init_kwargs["max_length"] = DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH

        # Explicit arguments from the constructor take precedence over kwargs
        resolved_device = infer_torch_device() if device is None else device
        init_kwargs["device"] = resolved_device
        if cache_dir:
            init_kwargs["cache_dir"] = cache_dir

        self._model = CrossEncoder(
            model_name=model,
            **init_kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerRerank"

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

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._model.predict(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = float(score)

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
