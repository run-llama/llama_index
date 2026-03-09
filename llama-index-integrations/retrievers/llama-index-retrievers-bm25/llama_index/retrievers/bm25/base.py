import json
import logging
import os

from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    build_metadata_filter_fn,
)

import bm25s
import Stemmer
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {
    "similarity_top_k": "similarity_top_k",
    "_verbose": "verbose",
    "corpus_weight_mask": "corpus_weight_mask",
}

DEFAULT_PERSIST_FILENAME = "retriever.json"


class BM25Retriever(BaseRetriever):
    r"""
    A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        stemmer (Stemmer.Stemmer, optional):
            The stemmer to use. Defaults to an english stemmer.
        language (str, optional):
            The language to use for stopword removal. Defaults to "en".
        existing_bm25 (bm25s.BM25, optional):
            An existing BM25 object to use. If not provided, nodes must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        token_pattern (str, optional):
            The token pattern to use. Defaults to (?u)\\b\\w\\w+\\b.
        skip_stemming (bool, optional):
            Whether to skip stemming. Defaults to False.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.

    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25s.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
        filters: Optional[MetadataFilters] = None,
        corpus_weight_mask: Optional[List[int]] = None,
    ) -> None:
        self.stemmer = stemmer or Stemmer.Stemmer("english")
        self.similarity_top_k = similarity_top_k
        self.token_pattern = token_pattern
        self.skip_stemming = skip_stemming

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            self.corpus = [
                node_to_metadata_dict(node) | {"node_id": node.node_id}
                for node in nodes
            ]

            corpus_tokens = bm25s.tokenize(
                [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
                stopwords=language,
                stemmer=self.stemmer if not skip_stemming else None,
                token_pattern=self.token_pattern,
                show_progress=verbose,
            )
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens, show_progress=verbose)

        if (
            self.bm25.scores.get("num_docs")
            and int(self.bm25.scores["num_docs"]) < self.similarity_top_k
        ):
            if int(self.bm25.scores["num_docs"]) == 0:
                raise ValueError(
                    "No nodes added to the retriever kindly add more data."
                )

            logger.warning(
                "As bm25s.BM25 requires k less than or equal to number of nodes added. Overriding the value of similarity_top_k to number of nodes added."
            )
            self.similarity_top_k = int(self.bm25.scores["num_docs"])

        self.corpus_weight_mask = corpus_weight_mask or None
        if filters and self.corpus:
            # Build a weight mask for each corpus to filter out only relevant nodes
            _corpus_dict = {
                corpus_token["node_id"]: corpus_token for corpus_token in self.corpus
            }
            _query_filter_fn = build_metadata_filter_fn(
                lambda node_id: _corpus_dict[node_id], filters
            )
            self.corpus_weight_mask = [
                int(_query_filter_fn(corpus_token["node_id"]))
                for corpus_token in self.corpus
            ]

            # Check if all nodes were filtered out
            if not any(self.corpus_weight_mask):
                raise ValueError(
                    "All nodes were filtered out by the metadata filters. "
                    "Please adjust your filters or add more data."
                )

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
