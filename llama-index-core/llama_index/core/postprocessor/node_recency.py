"""Node recency post-processor."""

from datetime import datetime
from typing import List, Optional, Set

import numpy as np

# NOTE: currently not being used
# DEFAULT_INFER_RECENCY_TMPL = (
#     "A question is provided.\n"
#     "The goal is to determine whether the question requires finding the most recent "
#     "context.\n"
#     "Please respond with YES or NO.\n"
#     "Question: What is the current status of the patient?\n"
#     "Answer: YES\n"
#     "Question: What happened in the Battle of Yorktown?\n"
#     "Answer: NO\n"
#     "Question: What are the most recent changes to the project?\n"
#     "Answer: YES\n"
#     "Question: How did Harry defeat Voldemort in the Battle of Hogwarts?\n"
#     "Answer: NO\n"
#     "Question: {query_str}\n"
#     "Answer: "
# )
# def parse_recency_pred(pred: str) -> bool:
#     """Parse recency prediction."""
#     if "YES" in pred:
#         return True
#     elif "NO" in pred:
#         return False
#     else:
#         raise ValueError(f"Invalid recency prediction: {pred}.")
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings


class FixedRecencyPostprocessor(BaseNodePostprocessor):
    """
    Fixed Recency post-processor.

    This post-processor does the following steps orders nodes by date.

    Assumes the date_key corresponds to a date field in the metadata.
    """

    top_k: int = 1
    date_key: str = "date"

    @classmethod
    def class_name(cls) -> str:
        return "FixedRecencyPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. Please install it with `pip install pandas`."
            )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        # sort nodes by date
        node_dates = pd.to_datetime(
            [node.node.metadata[self.date_key] for node in nodes]
        )
        sorted_node_idxs = np.flip(node_dates.argsort())
        sorted_nodes = [nodes[idx] for idx in sorted_node_idxs]

        return sorted_nodes[: self.top_k]


DEFAULT_QUERY_EMBEDDING_TMPL = (
    "The current document is provided.\n"
    "----------------\n"
    "{context_str}\n"
    "----------------\n"
    "Given the document, we wish to find documents that contain \n"
    "similar context. Note that these documents are older "
    "than the current document, meaning that certain details may be changed. \n"
    "However, the high-level context should be similar.\n"
)


class EmbeddingRecencyPostprocessor(BaseNodePostprocessor):
    """Embedding Recency post-processor."""

    embed_model: SerializeAsAny[BaseEmbedding] = Field(
        default_factory=lambda: Settings.embed_model
    )
    date_key: str = "date"
    similarity_cutoff: float = Field(default=0.7)
    query_embedding_tmpl: str = Field(default=DEFAULT_QUERY_EMBEDDING_TMPL)

    @classmethod
    def class_name(cls) -> str:
        return "EmbeddingRecencyPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. Please install it with `pip install pandas`."
            )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        # sort nodes by date
        node_dates = pd.to_datetime(
            [node.node.metadata[self.date_key] for node in nodes]
        )
        sorted_node_idxs = np.flip(node_dates.argsort())
        sorted_nodes: List[NodeWithScore] = [nodes[idx] for idx in sorted_node_idxs]

        # get embeddings for each node
        texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
        text_embeddings = self.embed_model.get_text_embedding_batch(texts=texts)

        node_ids_to_skip: Set[str] = set()
        for idx, node in enumerate(sorted_nodes):
            if node.node.node_id in node_ids_to_skip:
                continue
            # get query embedding for the "query" node
            # NOTE: not the same as the text embedding because
            # we want to optimize for retrieval results

            query_text = self.query_embedding_tmpl.format(
                context_str=node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            query_embedding = self.embed_model.get_query_embedding(query_text)

            for idx2 in range(idx + 1, len(sorted_nodes)):
                if sorted_nodes[idx2].node.node_id in node_ids_to_skip:
                    continue
                node2 = sorted_nodes[idx2]
                if (
                    np.dot(query_embedding, text_embeddings[idx2])
                    > self.similarity_cutoff
                ):
                    node_ids_to_skip.add(node2.node.node_id)

        return [
            node for node in sorted_nodes if node.node.node_id not in node_ids_to_skip
        ]


class TimeWeightedPostprocessor(BaseNodePostprocessor):
    """
    Time-weighted post-processor.

    Reranks a set of nodes based on their recency.

    """

    time_decay: float = Field(default=0.99)
    last_accessed_key: str = "__last_accessed__"
    time_access_refresh: bool = True
    # optionally set now (makes it easier to test)
    now: Optional[float] = None
    top_k: int = 1

    @classmethod
    def class_name(cls) -> str:
        return "TimeWeightedPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        now = self.now or datetime.now().timestamp()
        # TODO: refactor with get_top_k_embeddings

        similarities = []
        for node_with_score in nodes:
            # embedding similarity score
            score = node_with_score.score or 1.0
            node = node_with_score.node
            # time score
            if node.metadata is None:
                raise ValueError("metadata is None")

            last_accessed = node.metadata.get(self.last_accessed_key, None)
            if last_accessed is None:
                last_accessed = now

            hours_passed = (now - last_accessed) / 3600
            time_similarity = (1 - self.time_decay) ** hours_passed

            similarity = score + time_similarity

            similarities.append(similarity)

        sorted_tups = sorted(zip(similarities, nodes), key=lambda x: x[0], reverse=True)

        top_k = min(self.top_k, len(sorted_tups))
        result_tups = sorted_tups[:top_k]
        result_nodes = [
            NodeWithScore(node=n.node, score=score) for score, n in result_tups
        ]

        # set __last_accessed__ to now
        if self.time_access_refresh:
            for node_with_score in result_nodes:
                node_with_score.node.metadata[self.last_accessed_key] = now

        return result_nodes
