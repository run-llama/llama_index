import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device

dispatcher = get_dispatcher(__name__)


class ColPaliRerank(BaseNodePostprocessor):
    model: str = Field(description="Colpali model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cuda",
        description="Device to use for the model.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 5,
        model: str = "vidore/colpali-v1.2",
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
    ):
        device = infer_torch_device() if device is None else device
        super().__init__(
            top_n=top_n,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
            model=model,
        )

        self._model = ColPali.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(model)

    @classmethod
    def class_name(cls) -> str:
        return "ColPaliRerank"

    def get_image_paths(self, nodes: List[NodeWithScore]):
        image_paths = []
        for node_ in nodes:
            image_paths.append(node_.node.metadata["file_path"])

        return image_paths

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)

    def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        images = []
        for image_path in image_paths:
            images.append(self.load_image(image_path))

        return images

    def _calculate_sim(self, query: str, images_paths: List[str]) -> List[float]:
        # Load the images
        images = self.load_images(images_paths)

        # Process the inputs
        batch_images = self._processor.process_images(images).to(self._model.device)
        batch_queries = self._processor.process_queries([query]).to(self._model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = self._model(**batch_images)
            querry_embeddings = self._model(**batch_queries)

        return self._processor.score_multi_vector(querry_embeddings, image_embeddings)

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

        image_paths = self.get_image_paths(nodes)

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._calculate_sim(query_bundle.query_str, image_paths)[
                0
            ].tolist()

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = float(score)

            reranked_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: reranked_nodes})

        dispatcher.event(ReRankEndEvent(nodes=reranked_nodes))
        return reranked_nodes
