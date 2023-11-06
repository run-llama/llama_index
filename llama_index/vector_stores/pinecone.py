"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

ID_KEY = "id"
VECTOR_KEY = "values"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"

DEFAULT_BATCH_SIZE = 100

_logger = logging.getLogger(__name__)


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
    return build_dict(inputs)


def get_default_tokenizer() -> Callable:
    """Get default tokenizer.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    from transformers import BertTokenizerFast

    orig_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # set some default arguments, so input is just a list of strings
    return partial(
        orig_tokenizer,
        padding=True,
        truncation=True,
        max_length=512,
    )


def _to_pinecone_filter(standard_filters: MetadataFilters) -> dict:
    """Convert from standard dataclass to pinecone filter dict."""
    filters = {}
    for filter in standard_filters.filters:
        filters[filter.key] = filter.value
    return filters


import_err_msg = (
    "`pinecone` package not found, please run `pip install pinecone-client`"
)


class PineconeVectorStore(BasePydanticVectorStore):
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
    flat_metadata: bool = False

    api_key: Optional[str]
    index_name: Optional[str]
    environment: Optional[str]
    namespace: Optional[str]
    insert_kwargs: Optional[Dict]
    add_sparse_vector: bool
    text_key: str
    batch_size: int
    remove_text_from_metadata: bool

    _pinecone_index: Any = PrivateAttr()
    _tokenizer: Optional[Callable] = PrivateAttr()

    def __init__(
        self,
        pinecone_index: Optional[Any] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        insert_kwargs: Optional[Dict] = None,
        add_sparse_vector: bool = False,
        tokenizer: Optional[Callable] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        remove_text_from_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        try:
            import pinecone
        except ImportError:
            raise ImportError(import_err_msg)

        if pinecone_index is not None:
            self._pinecone_index = cast(pinecone.Index, pinecone_index)
        else:
            if index_name is None or environment is None:
                raise ValueError(
                    "Must specify index_name and environment "
                    "if not directly passing in client."
                )

            pinecone.init(api_key=api_key, environment=environment)
            self._pinecone_index = pinecone.Index(index_name)

        insert_kwargs = insert_kwargs or {}

        if tokenizer is None and add_sparse_vector:
            tokenizer = get_default_tokenizer()
        self._tokenizer = tokenizer

        super().__init__(
            index_name=index_name,
            environment=environment,
            api_key=api_key,
            namespace=namespace,
            insert_kwargs=insert_kwargs,
            add_sparse_vector=add_sparse_vector,
            text_key=text_key,
            batch_size=batch_size,
            remove_text_from_metadata=remove_text_from_metadata,
        )

    @classmethod
    def from_params(
        cls,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        insert_kwargs: Optional[Dict] = None,
        add_sparse_vector: bool = False,
        tokenizer: Optional[Callable] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        remove_text_from_metadata: bool = False,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        try:
            import pinecone
        except ImportError:
            raise ImportError(import_err_msg)

        pinecone.init(api_key=api_key, environment=environment)
        pinecone_index = pinecone.Index(index_name)

        return cls(
            pinecone_index=pinecone_index,
            api_key=api_key,
            index_name=index_name,
            environment=environment,
            namespace=namespace,
            insert_kwargs=insert_kwargs,
            add_sparse_vector=add_sparse_vector,
            tokenizer=tokenizer,
            text_key=text_key,
            batch_size=batch_size,
            remove_text_from_metadata=remove_text_from_metadata,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PinconeVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = []
        entries = []
        for node in nodes:
            node_id = node.node_id

            metadata = node_to_metadata_dict(
                node,
                remove_text=self.remove_text_from_metadata,
                flat_metadata=self.flat_metadata,
            )

            entry = {
                ID_KEY: node_id,
                VECTOR_KEY: node.get_embedding(),
                METADATA_KEY: metadata,
            }
            if self.add_sparse_vector and self._tokenizer is not None:
                sparse_vector = generate_sparse_vectors(
                    [node.get_content(metadata_mode=MetadataMode.EMBED)],
                    self._tokenizer,
                )[0]
                entry[SPARSE_VECTOR_KEY] = sparse_vector

            ids.append(node_id)
            entries.append(entry)
        self._pinecone_index.upsert(
            entries,
            namespace=self.namespace,
            batch_size=self.batch_size,
            **self.insert_kwargs,
        )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": ref_doc_id}},
            namespace=self.namespace,
            **delete_kwargs,
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
        if (
            query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            and self._tokenizer is not None
        ):
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
            if "filter" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for pinecone specific items that are "
                    "not supported via the generic query interface."
                )
            filter = _to_pinecone_filter(query.filters)
        else:
            filter = kwargs.pop("filter", {})

        response = self._pinecone_index.query(
            vector=query_embedding,
            sparse_vector=sparse_vector,
            top_k=query.similarity_top_k,
            include_values=True,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter,
            **kwargs,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for match in response.matches:
            try:
                node = metadata_dict_to_node(match.metadata)
                node.embedding = match.values
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                _logger.debug(
                    "Failed to parse Node metadata, fallback to legacy logic."
                )
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    match.metadata, text_key=self.text_key
                )

                text = match.metadata[self.text_key]
                id = match.id
                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            top_k_ids.append(match.id)
            top_k_nodes.append(node)
            top_k_scores.append(match.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
