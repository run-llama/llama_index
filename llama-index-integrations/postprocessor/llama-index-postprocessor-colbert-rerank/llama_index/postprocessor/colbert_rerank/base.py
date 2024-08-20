from typing import Any, List, Optional

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

import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_COLBERT_MAX_LENGTH = 512

dispatcher = get_dispatcher(__name__)


class ColbertRerank(BaseNodePostprocessor):
    model: str = Field(description="Colbert model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 5,
        model: str = "colbert-ir/colbertv2.0",
        tokenizer: str = "colbert-ir/colbertv2.0",
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
    ):
        device = infer_torch_device() if device is None else device
        super().__init__(
            top_n=top_n,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._model = AutoModel.from_pretrained(model)

    @classmethod
    def class_name(cls) -> str:
        return "ColbertRerank"

    def _calculate_sim(self, query: str, documents_text_list: List[str]) -> List[float]:
        # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
        # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
        query_encoding = self._tokenizer(query, return_tensors="pt")
        query_embedding = self._model(**query_encoding).last_hidden_state
        rerank_score_list = []

        for document_text in documents_text_list:
            document_encoding = self._tokenizer(
                document_text, return_tensors="pt", truncation=True, max_length=512
            )
            document_embedding = self._model(**document_encoding).last_hidden_state

            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
            )

            # Take the maximum similarity for each query token (across all document tokens)
            # sim_matrix shape: [batch_size, query_length, doc_length]
            max_sim_scores, _ = torch.max(sim_matrix, dim=2)
            rerank_score_list.append(torch.mean(max_sim_scores, dim=1))

        return rerank_score_list

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

        nodes_text_list = [
            str(node.node.get_content(metadata_mode=MetadataMode.EMBED))
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
            scores = self._calculate_sim(query_bundle.query_str, nodes_text_list)

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
