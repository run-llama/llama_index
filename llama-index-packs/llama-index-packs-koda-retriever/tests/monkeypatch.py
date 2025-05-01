"""
This is a monkeypatch script that will allow us to SIMULATE hybrid search functionality to the SimpleVectorStore class.
Emphasis on simulate, as we are not actually adding the functionality to the class, but rather
just allowing the class to accept VectorStoreQuery objects with the mode set to VectorStoreQueryMode.HYBRID.

This also flows upstream to the VectorStoreIndex class - which will default use SimpleVectorStore index if you don't provide
your own Vector Store. This is why we are monkeypatching the SimpleVectorStore class and not the VectorStoreIndex class.

I didn't write MOST of this code, I just copied it from the llama_index library and modified it to fit my needs.

AUTHOR: no_dice
"""

import logging
from typing import Any, Callable, List, Mapping, Optional, cast
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.readers import StringIterableReader

logger = logging.getLogger(__name__)

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}

MMR_MODE = VectorStoreQueryMode.MMR

NAMESPACE_SEP = "__"
DEFAULT_VECTOR_STORE = "default"

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}


def _build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[str], Mapping[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[str], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.legacy_filters() if metadata_filters else []
    if not filter_list:
        return lambda _: True

    def filter_fn(node_id: str) -> bool:
        metadata = metadata_lookup_fn(node_id)
        for filter_ in filter_list:
            metadata_value = metadata.get(filter_.key, None)
            if metadata_value is None:
                return False
            elif isinstance(metadata_value, list):
                if filter_.value not in metadata_value:
                    return False
            elif isinstance(metadata_value, (int, float, str, bool)):
                if metadata_value != filter_.value:
                    return False
        return True

    return filter_fn


def monkey_query(
    self,
    query: VectorStoreQuery,
    **kwargs: Any,
):
    """Get nodes for response."""
    # Prevent metadata filtering on stores that were persisted without metadata.
    if (
        query.filters is not None
        and self._data.embedding_dict
        and not self._data.metadata_dict
    ):
        raise ValueError(
            "Cannot filter stores that were persisted without metadata. "
            "Please rebuild the store with metadata to enable filtering."
        )
    # Prefilter nodes based on the query filter and node ID restrictions.
    query_filter_fn = _build_metadata_filter_fn(
        lambda node_id: self._data.metadata_dict[node_id], query.filters
    )

    if query.node_ids is not None:
        available_ids = set(query.node_ids)

        def node_filter_fn(node_id: str) -> bool:
            return node_id in available_ids

    else:

        def node_filter_fn(node_id: str) -> bool:
            return True

    node_ids = []
    embeddings = []
    # TODO: consolidate with get_query_text_embedding_similarities
    for node_id, embedding in self._data.embedding_dict.items():
        if node_filter_fn(node_id) and query_filter_fn(node_id):
            node_ids.append(node_id)
            embeddings.append(embedding)

    query_embedding = cast(List[float], query.query_embedding)

    if query.mode in LEARNER_MODES:
        top_similarities, top_ids = get_top_k_embeddings_learner(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
        )
    elif query.mode == MMR_MODE:
        mmr_threshold = kwargs.get("mmr_threshold")
        top_similarities, top_ids = get_top_k_mmr_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
            mmr_threshold=mmr_threshold,
        )
    elif query.mode == VectorStoreQueryMode.DEFAULT:
        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
        )
    elif query.mode == VectorStoreQueryMode.HYBRID:  # where I made my changes
        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
        )
    else:
        raise ValueError(f"Invalid query mode: {query.mode}")

    return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)


SimpleVectorStore.query = (
    monkey_query  # very important, this is where the monkeypatching happens
)


def monkey_patch_vector_store_index() -> VectorStoreIndex:  # I did write this part..
    """Returns a monkey-patched vector store index to simulate hybrid retrieval behavior."""
    import sys

    print(sys.executable)

    loader = StringIterableReader()

    documents = loader.load_data(
        texts=[
            "This is a test document.",
            "This is another test document.",
            "This is a third test document.",
        ]
    )

    return VectorStoreIndex.from_documents(documents)
