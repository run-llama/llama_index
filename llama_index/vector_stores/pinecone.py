"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
import os
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (MetadataFilters,
                                             NodeWithEmbedding, VectorStore,
                                             VectorStoreQuery,
                                             VectorStoreQueryMode,
                                             VectorStoreQueryResult)

_logger = logging.getLogger(__name__)


def get_metadata_from_node_info(
    node_info: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get metadata from node extra info."""
    metadata = {}
    for key, value in node_info.items():
        metadata[field_prefix + "_" + key] = value
    return metadata


def get_node_info_from_metadata(
    metadata: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get node extra info from metadata."""
    node_extra_info = {}
    for key, value in metadata.items():
        if key.startswith(field_prefix + "_"):
            node_extra_info[key.replace(field_prefix + "_", "")] = value
    return node_extra_info


def build_dict(input_batch: List[List[int]]) -> List[Dict[str, Any]]:
    """Build a list of sparse dictionaries from a batch of input_ids.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    # store a batch of sparse embeddings
    sparse_emb = []
    # iterate through input batch
    for token_ids in input_batch:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(float(d[idx]))
        sparse_emb.append({"indices": indices, "values": values})
    # return sparse_emb list
    return sparse_emb


def generate_sparse_vectors(
    context_batch: List[str], tokenizer: Callable
) -> List[Dict[str, Any]]:
    """Generate sparse vectors from a batch of contexts.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    # create batch of input_ids
    inputs = tokenizer(context_batch)["input_ids"]
    # create sparse dictionaries
    sparse_embeds = build_dict(inputs)
    return sparse_embeds


def get_default_tokenizer() -> Callable:
    """Get default tokenizer.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    from transformers import BertTokenizerFast

    orig_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # set some default arguments, so input is just a list of strings
    tokenizer = partial(
        orig_tokenizer,
        padding=True,
        truncation=True,
        max_length=512,
    )
    return tokenizer


def _to_pinecone_filter(standard_filters: MetadataFilters) -> dict:
    """Convert from standard dataclass to pinecone filter dict."""
    filters = {}
    for filter in standard_filters.filters:
        filters[filter.key] = filter.value
    return filter


class PineconeVectorStore(VectorStore):
    """Pinecone Vector Store.

    In this vector store, embeddings and docs are stored within a
    Pinecone index.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes.

    Args:
        pinecone_index (Optional[pinecone.Index]): Pinecone index instance
        insert_kwargs (Optional[Dict]): insert kwargs during `upsert` call.
        add_sparse_vector (bool): whether to add sparse vector to index.
        tokenizer (Optional[Callable]): tokenizer to use to generate sparse

    """

    stores_text: bool = True

    def __init__(
        self,
        pinecone_index: Optional[Any] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        insert_kwargs: Optional[Dict] = None,
        add_sparse_vector: bool = False,
        tokenizer: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`pinecone` package not found, please run `pip install pinecone-client`"
        )
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._index_name = index_name
        self._environment = environment
        self._namespace = namespace
        if pinecone_index is not None:
            self._pinecone_index = cast(pinecone.Index, pinecone_index)
        else:
            if "PINECONE_API_KEY" not in os.environ:
                raise ValueError(
                    "Must specify PINECONE_API_KEY via env variable "
                    "if not directly passing in client."
                )
            if index_name is None or environment is None:
                raise ValueError(
                    "Must specify index_name and environment "
                    "if not directly passing in client."
                )

            pinecone.init(environment=environment)
            self._pinecone_index = pinecone.Index(index_name)

        self._insert_kwargs = insert_kwargs or {}

        self._add_sparse_vector = add_sparse_vector
        if tokenizer is None:
            tokenizer = get_default_tokenizer()
        self._tokenizer = tokenizer

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        ids = []
        for result in embedding_results:
            node_id = result.id
            node = result.node

            metadata = {
                "text": node.get_text(),
                # NOTE: this is the reference to source doc
                "doc_id": node.ref_doc_id,
                "id": node_id,
            }
            if node.extra_info:
                # TODO: check if overlap with default metadata keys
                metadata.update(
                    get_metadata_from_node_info(node.extra_info, "extra_info")
                )
            if node.node_info:
                # TODO: check if overlap with default metadata keys
                metadata.update(
                    get_metadata_from_node_info(node.node_info, "node_info")
                )
            # if additional metadata keys overlap with the default keys,
            # then throw an error

            entry = {
                "id": node_id,
                "values": result.embedding,
                "metadata": metadata,
            }
            if self._add_sparse_vector:
                sparse_vector = generate_sparse_vectors(
                    [node.get_text()], self._tokenizer
                )[0]
                entry.update({"sparse_values": sparse_vector})
            self._pinecone_index.upsert(
                [entry], namespace=self._namespace, **self._insert_kwargs
            )
            ids.append(node_id)
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}, **delete_kwargs
        )

    @property
    def client(self) -> Any:
        """Return Pinecone client."""
        return self._pinecone_index

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        sparse_vector = None
        if query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID):
            if query.query_str is None:
                raise ValueError(
                    "query_str must be specified if mode is SPARSE or HYBRID."
                )
            sparse_vector = generate_sparse_vectors([query.query_str], self._tokenizer)[
                0
            ]
            if query.alpha is not None:
                sparse_vector = {
                    "indices": sparse_vector["indices"],
                    "values": [v * (1 - query.alpha) for v in sparse_vector["values"]],
                }

        query_embedding = None
        if query.mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID):
            query_embedding = cast(List[float], query.query_embedding)
            if query.alpha is not None:
                query_embedding = [v * query.alpha for v in query_embedding]
            
        if query.filters is not None:
            if 'filter' in kwargs:
                raise ValueError("Cannot specify filter via both query and kwargs. "
                                 "Use kwargs only for pinecone specific items that are "
                                 "not supported via the generic query interface.")
            filter = _to_pinecone_filter(query.filters)
        else:
            filter = kwargs.pop('filter', {})


        response = self._pinecone_index.query(
            vector=query_embedding,
            sparse_vector=sparse_vector,
            top_k=query.similarity_top_k,
            include_values=True,
            include_metadata=True,
            namespace=self._namespace,
            filter=filter,
            **kwargs,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for match in response.matches:
            text = match.metadata["text"]
            extra_info = get_node_info_from_metadata(match.metadata, "extra_info")
            node_info = get_node_info_from_metadata(match.metadata, "node_info")
            doc_id = match.metadata["doc_id"]
            id = match.metadata["id"]

            node = Node(
                text=text,
                extra_info=extra_info,
                node_info=node_info,
                doc_id=id,
                relationships={DocumentRelationship.SOURCE: doc_id},
            )
            top_k_ids.append(match.id)
            top_k_nodes.append(node)
            top_k_scores.append(match.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
