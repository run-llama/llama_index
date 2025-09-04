import os
from typing import List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)

# Maximum cumulative word count per batch.
MAX_WORDS_PER_BATCH = 10000


class AIMonRerank(BaseNodePostprocessor):
    model: str = Field(description="AIMon's reranking model name.")
    top_n: int = Field(description="Top N nodes to return.")
    task_definition: str = Field(
        default="Determine the relevance of context documents with respect to the user query.",
        description="The task definition for the AIMon reranker.",
    )

    _api_key: str = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "retrieval_relevance",
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
                "Must pass in AIMon API key or specify via AIMON_API_KEY environment variable"
            )
        try:
            from aimon import Client
        except ImportError:
            raise ImportError(
                "Cannot import AIMon package, please `pip install aimon`."
            )

        self._client = Client(auth_header=f"Bearer {api_key}")

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
        if len(nodes) == 0:
            return []

        # Extract text content from each node.
        texts = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
        ]

        # Build batches where the total number of words is <= MAX_WORDS_PER_BATCH.
        batches = []  # List of (batch_texts, corresponding indices)
        current_batch = []  # List of texts for the current batch.
        current_indices = []  # Corresponding indices of texts in the original list.
        current_word_count = 0

        for i, text in enumerate(texts):
            word_count = len(text.split())
            if current_word_count + word_count <= MAX_WORDS_PER_BATCH:
                current_batch.append(text)
                current_indices.append(i)
                current_word_count += word_count
            else:
                batches.append((current_batch, current_indices))
                # Start a new batch with the current text.
                current_batch = [text]
                current_indices = [i]
                current_word_count = word_count
        if current_batch:
            batches.append((current_batch, current_indices))

        # Prepare a list to hold scores for each text in the original order.
        all_scores = [None] * len(texts)

        for batch_num, (batch_texts, indices) in enumerate(batches, start=1):
            scores_batch = self._client.retrieval.rerank(
                context_docs=batch_texts,
                queries=[query_bundle.query_str],
                task_definition=self.task_definition,
            )

            batch_scores = scores_batch[0]
            for idx, score in zip(indices, batch_scores):
                all_scores[idx] = score

        normalized_scores = [score / 100 for score in all_scores]

        # Attach scores to nodes.
        scored_nodes = [
            NodeWithScore(node=nodes[i].node, score=normalized_scores[i])
            for i in range(len(nodes))
        ]

        # Sort nodes by score in descending order and keep the top N.
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        new_nodes = scored_nodes[: self.top_n]

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
