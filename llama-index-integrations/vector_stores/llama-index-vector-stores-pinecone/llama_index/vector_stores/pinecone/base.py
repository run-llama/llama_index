"""
Pinecone Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
import pinecone
from llama_index.vector_stores.pinecone.utils import (
    _import_pinecone,
    _is_pinecone_v3,
    DefaultPineconeSparseEmbedding,
)

ID_KEY = "id"
VECTOR_KEY = "values"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"

DEFAULT_BATCH_SIZE = 100

_logger = logging.getLogger(__name__)


def _transform_pinecone_filter_condition(condition: str) -> str:
    """Translate standard metadata filter op to Pinecone specific spec."""
    if condition == "and":
        return "$and"
    elif condition == "or":
        return "$or"
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_pinecone_filter_operator(operator: str) -> str:
    """Translate standard metadata filter operator to Pinecone specific spec."""
    if operator == "!=":
        return "$ne"
    elif operator == "==":
        return "$eq"
    elif operator == ">":
        return "$gt"
    elif operator == "<":
        return "$lt"
    elif operator == ">=":
        return "$gte"
    elif operator == "<=":
        return "$lte"
    elif operator == "in":
        return "$in"
    elif operator == "nin":
        return "$nin"
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_pinecone_filter(standard_filters: MetadataFilters) -> dict:
    """Convert from standard dataclass to pinecone filter dict."""
    filters = {}
    filters_list = []
    condition = standard_filters.condition or "and"
    condition = _transform_pinecone_filter_condition(condition)
    if standard_filters.filters:
        for filter in standard_filters.filters:
            if isinstance(filter, MetadataFilters):
                sub_filter = _to_pinecone_filter(filter)
                if sub_filter:
                    filters_list.append(sub_filter)
            elif filter.operator:
                filters_list.append(
                    {
                        filter.key: {
                            _transform_pinecone_filter_operator(
                                filter.operator
                            ): filter.value
                        }
                    }
                )
            else:
                filters_list.append({filter.key: filter.value})

    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]
    elif len(filters_list) > 1:
        filters[condition] = filters_list
    return filters


import_err_msg = (
    "`pinecone` package not found, please run `pip install pinecone-client`"
)


class PineconeVectorStore(BasePydanticVectorStore):
    """
    Pinecone Vector Store.

    In this vector store, embeddings and docs are stored within a
    Pinecone index.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes.

    Args:
        pinecone_index (Optional[Union[pinecone.Pinecone.Index, pinecone.Index]]): Pinecone index instance,
        pinecone.Pinecone.Index for clients >= 3.0.0; pinecone.Index for older clients.
        insert_kwargs (Optional[Dict]): insert kwargs during `upsert` call.
        add_sparse_vector (bool): whether to add sparse vector to index.
        tokenizer (Optional[Callable]): tokenizer to use to generate sparse
        default_empty_query_vector (Optional[List[float]]): default empty query vector.
            Defaults to None. If not None, then this vector will be used as the query
            vector if the query is empty.

    Examples:
        `pip install llama-index-vector-stores-pinecone`

        ```python
        import os
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        # Set up Pinecone API key
        os.environ["PINECONE_API_KEY"] = "<Your Pinecone API key, from app.pinecone.io>"
        api_key = os.environ["PINECONE_API_KEY"]

        # Create Pinecone Vector Store
        pc = Pinecone(api_key=api_key)

        pc.create_index(
            name="quickstart",
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

        pinecone_index = pc.Index("quickstart")

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        ```

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

    _pinecone_index: pinecone.Index = PrivateAttr()
    _sparse_embedding_model: Optional[BaseSparseEmbedding] = PrivateAttr()

    def __init__(
        self,
        pinecone_index: Optional[
            Any
        ] = None,  # Dynamic import prevents specific type hinting here
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
        default_empty_query_vector: Optional[List[float]] = None,
        sparse_embedding_model: Optional[BaseSparseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        insert_kwargs = insert_kwargs or {}

        if add_sparse_vector:
            if sparse_embedding_model is not None:
                sparse_embedding_model = sparse_embedding_model
            elif tokenizer is not None:
                sparse_embedding_model = DefaultPineconeSparseEmbedding(
                    tokenizer=tokenizer
                )
            else:
                sparse_embedding_model = DefaultPineconeSparseEmbedding()
        else:
            sparse_embedding_model = None

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

        self._sparse_embedding_model = sparse_embedding_model

        # TODO: Make following instance check stronger -- check if pinecone_index is not pinecone.Index, else raise
        #  ValueError
        if isinstance(pinecone_index, str):
            raise ValueError(
                "`pinecone_index` cannot be of type `str`; should be an instance of pinecone.Index, "
            )

        self._pinecone_index = pinecone_index or self._initialize_pinecone_client(
            api_key, index_name, environment, **kwargs
        )

    @classmethod
    def _initialize_pinecone_client(
        cls,
        api_key: Optional[str],
        index_name: Optional[str],
        environment: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """
        Initialize Pinecone client based on version.

        If client version <3.0.0, use pods-based initialization; else, use serverless initialization.
        """
        if not index_name:
            raise ValueError(
                "`index_name` is required for Pinecone client initialization"
            )

        pinecone = _import_pinecone()

        if (
            not _is_pinecone_v3()
        ):  # If old version of Pinecone client (version bifurcation temporary):
            if not environment:
                raise ValueError("environment is required for Pinecone client < 3.0.0")
            pinecone.init(api_key=api_key, environment=environment)
            return pinecone.Index(index_name)
        else:  # If new version of Pinecone client (serverless):
            pinecone_instance = pinecone.Pinecone(
                api_key=api_key, source_tag="llamaindex"
            )
            return pinecone_instance.Index(index_name)

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
        default_empty_query_vector: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        pinecone_index = cls._initialize_pinecone_client(
            api_key, index_name, environment, **kwargs
        )

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
            default_empty_query_vector=default_empty_query_vector,
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
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = []
        entries = []
        sparse_inputs = []
        for node in nodes:
            node_id = node.node_id

            metadata = node_to_metadata_dict(
                node,
                remove_text=self.remove_text_from_metadata,
                flat_metadata=self.flat_metadata,
            )

            if self.add_sparse_vector and self._sparse_embedding_model is not None:
                sparse_inputs.append(node.get_content(metadata_mode=MetadataMode.EMBED))

            if node.ref_doc_id is not None:
                node_id = f"{node.ref_doc_id}#{node_id}"

            ids.append(node_id)

            entry = {
                ID_KEY: node_id,
                VECTOR_KEY: node.get_embedding(),
                METADATA_KEY: metadata,
            }
            entries.append(entry)

        # batch sparse embedding generation
        if sparse_inputs:
            sparse_vectors = self._sparse_embedding_model.get_text_embedding_batch(
                sparse_inputs
            )
            for i, sparse_vector in enumerate(sparse_vectors):
                entries[i][SPARSE_VECTOR_KEY] = {
                    "indices": list(sparse_vector.keys()),
                    "values": list(sparse_vector.values()),
                }

        self._pinecone_index.upsert(
            entries,
            namespace=self.namespace,
            batch_size=self.batch_size,
            **self.insert_kwargs,
        )
        return ids

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[List[MetadataFilters]] = None,
        limit: int = 100,
        include_values: bool = False,
    ) -> List[BaseNode]:
        filter = None
        if filters is not None:
            filter = _to_pinecone_filter(filters)

        if node_ids is not None:
            raise ValueError(
                "Getting nodes by node id not supported by Pinecone at the time of writing."
            )

        if node_ids is None and filters is None:
            raise ValueError("Filters must be specified")

        # Pinecone requires a query vector, so default to 0s if not provided
        query_vector = [0.0] * self._pinecone_index.describe_index_stats()["dimension"]

        response = self._pinecone_index.query(
            top_k=limit,
            vector=query_vector,
            namespace=self.namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=True,
        )

        nodes = [metadata_dict_to_node(match.metadata) for match in response.matches]
        if include_values:
            for node, match in zip(nodes, response.matches):
                node.embedding = match.values

        return nodes

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        try:
            # delete by filtering on the doc_id metadata
            self._pinecone_index.delete(
                filter={"doc_id": {"$eq": ref_doc_id}},
                namespace=self.namespace,
                **delete_kwargs,
            )
        except Exception:
            # fallback to deleting by prefix for serverless
            # TODO: this is a bit of a hack, we should find a better way to handle this
            id_gen = self._pinecone_index.list(
                prefix=ref_doc_id, namespace=self.namespace
            )
            ids_to_delete = list(id_gen)
            if ids_to_delete:
                self._pinecone_index.delete(
                    ids=ids_to_delete, namespace=self.namespace, **delete_kwargs
                )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Deletes nodes using their ids.

        Args:
            node_ids (Optional[List[str]], optional): List of node IDs. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        """
        node_ids = node_ids or []

        if filters is not None:
            filter = _to_pinecone_filter(filters)
        else:
            filter = None

        self._pinecone_index.delete(
            ids=node_ids, namespace=self.namespace, filter=filter, **delete_kwargs
        )

    def clear(self) -> None:
        """Clears the index."""
        self._pinecone_index.delete(namespace=self.namespace, delete_all=True)

    @property
    def client(self) -> Any:
        """Return Pinecone client."""
        return self._pinecone_index

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        pinecone_sparse_vector = None
        if (
            query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            and self._sparse_embedding_model is not None
        ):
            if query.query_str is None:
                raise ValueError(
                    "query_str must be specified if mode is SPARSE or HYBRID."
                )
            sparse_vector = self._sparse_embedding_model.get_query_embedding(
                query.query_str
            )
            if query.alpha is not None:
                pinecone_sparse_vector = {
                    "indices": list(sparse_vector.keys()),
                    "values": [v * (1 - query.alpha) for v in sparse_vector.values()],
                }
            else:
                pinecone_sparse_vector = {
                    "indices": list(sparse_vector.keys()),
                    "values": list(sparse_vector.values()),
                }

        # pinecone requires a query embedding, so default to 0s if not provided
        if query.query_embedding is not None:
            dimension = len(query.query_embedding)
        else:
            dimension = self._pinecone_index.describe_index_stats()["dimension"]
        query_embedding = [0.0] * dimension

        if query.mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID):
            query_embedding = cast(List[float], query.query_embedding)
            if query.alpha is not None:
                query_embedding = [v * query.alpha for v in query_embedding]

        if query.filters is not None:
            if "filter" in kwargs or "pinecone_query_filters" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for pinecone specific items that are "
                    "not supported via the generic query interface."
                )
            filter = _to_pinecone_filter(query.filters)
        elif "pinecone_query_filters" in kwargs:
            filter = kwargs.pop("pinecone_query_filters")
        else:
            filter = kwargs.pop("filter", {})

        response = self._pinecone_index.query(
            vector=query_embedding,
            sparse_vector=pinecone_sparse_vector,
            top_k=query.similarity_top_k,
            include_values=kwargs.pop("include_values", True),
            include_metadata=kwargs.pop("include_metadata", True),
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
