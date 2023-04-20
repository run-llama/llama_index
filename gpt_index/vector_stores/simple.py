"""Simple vector store index."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast, Callable, Tuple
from gpt_index.embeddings.base import similarity as default_similarity_fn
from functools import partial
from datetime import datetime

from dataclasses_json import DataClassJsonMixin

from gpt_index.indices.query.embedding_utils import get_top_k_embeddings
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
    VectorStoreQueryConfig,
)
from gpt_index.embeddings.base import similarity as default_similarity_fn
from gpt_index.data_structs.node_v2 import Node


def time_weighted_similarity_fn(
    time_decay_rate: float,
    hours_passed: int,
    query_embedding: List[float],
    embedding: List[float],
) -> float:
    """Time weighted similarity function."""
    semantic_similarity = default_similarity_fn(query_embedding, embedding)
    time_similarity = (1 - time_decay_rate) ** hours_passed
    return semantic_similarity + time_similarity


def get_top_k_nodes(
    query_embedding: List[float],
    nodes: List[Node],
    node_embeddings: List[List[float]],
    embed_similarity_fn: Optional[Callable[..., float]] = None,
    query_config: Optional[VectorStoreQueryConfig] = None,
    similarity_cutoff: Optional[float] = None,
    now: Optional[float] = None,
) -> Tuple[List[float], List]:
    """Get top k node ids by similarity.

    More general version of get_top_k_embeddings in
    gpt_index.indices.query.embedding_utils. Allows for time decay.

    """
    now = now or datetime.now().timestamp()
    # TODO: refactor with get_top_k_embeddings

    query_config = query_config or VectorStoreQueryConfig()
    similarity_fn = embed_similarity_fn or default_similarity_fn
    similarities = []
    for idx, node in enumerate(nodes):
        node_embedding = node_embeddings[idx]
        embed_similarity = similarity_fn(query_embedding, node_embedding)
        if node.node_info is None:
            raise ValueError("node_info is None")

        last_accessed = node.node_info.get("__last_accessed__", None)
        if last_accessed is None:
            last_accessed = now

        hours_passed = (now - last_accessed) / 3600
        time_similarity = (1 - query_config.time_decay_rate) ** hours_passed

        similarity = embed_similarity + time_similarity

        similarities.append(similarity)

    sorted_tups = sorted(zip(similarities, nodes), key=lambda x: x[0], reverse=True)

    if similarity_cutoff is not None:
        sorted_tups = [tup for tup in sorted_tups if tup[0] > similarity_cutoff]

    similarity_top_k = query_config.similarity_top_k or len(sorted_tups)
    result_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in result_tups]
    result_nodes = [n for _, n in result_tups]

    # set __last_accessed__ to now
    if query_config.time_access_refresh:
        for node in result_nodes:
            node.get_node_info()["__last_accessed__"] = now

    result_ids = [n.get_doc_id() for n in result_nodes]

    return result_similarities, result_ids


@dataclass
class SimpleVectorStoreData(DataClassJsonMixin):
    """Simple Vector Store Data container.

    Args:
        embedding_dict (Optional[dict]): dict mapping doc_ids to embeddings.
        text_id_to_doc_id (Optional[dict]): dict mapping text_ids to doc_ids.

    """

    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_doc_id: Dict[str, str] = field(default_factory=dict)


class SimpleVectorStore(VectorStore):
    """Simple Vector Store.

    In this vector store, embeddings are stored within a simple, in-memory dictionary.

    Args:
        simple_vector_store_data_dict (Optional[dict]): data dict
            containing the embeddings and doc_ids. See SimpleVectorStoreData
            for more details.
    """

    stores_text: bool = False

    def __init__(
        self,
        simple_vector_store_data_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if simple_vector_store_data_dict is None:
            self._data = SimpleVectorStoreData()
        else:
            self._data = SimpleVectorStoreData.from_dict(simple_vector_store_data_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimpleVectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    @property
    def config_dict(self) -> dict:
        """Get config dict."""
        return {
            "simple_vector_store_data_dict": self._data.to_dict(),
        }

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self._data.embedding_dict[text_id]

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding_results to index."""
        for result in embedding_results:
            text_id = result.id
            self._data.embedding_dict[text_id] = result.embedding
            self._data.text_id_to_doc_id[text_id] = result.doc_id
        return [result.id for result in embedding_results]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        text_ids_to_delete = set()
        for text_id, doc_id_ in self._data.text_id_to_doc_id.items():
            if doc_id == doc_id_:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            del self._data.embedding_dict[text_id]
            del self._data.text_id_to_doc_id[text_id]

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._data.embedding_dict.items()
        node_ids = [t[0] for t in items]
        embeddings = [t[1] for t in items]

        query_embedding = cast(List[float], query.query_embedding)
        if query.query_config.use_time_decay:
            if query.docstore is None:
                raise ValueError("query.docstore cannot be None")
            nodes = query.docstore.get_nodes(node_ids)
            top_similarities, top_ids = get_top_k_nodes(
                query_embedding,
                nodes,
                embeddings,
                query_config=query.query_config,
            )
        else:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.query_config.similarity_top_k,
                embedding_ids=node_ids,
            )

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
