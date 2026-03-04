"""DashVector Vector Store."""

import logging
from typing import Any, List, Optional, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from dashvector import Doc

DEFAULT_BATCH_SIZE = 100
logger = logging.getLogger(__name__)


def _to_dashvector_filter(
    standard_filters: Optional[MetadataFilters] = None,
) -> Optional[str]:
    """Convert from standard filter to dashvector filter dict."""
    if standard_filters is None:
        return None

    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            value = f"'{filter.value}'"
        else:
            value = f"{filter.value}"
        filters.append(f"{filter.key} = {value}")
    return " and ".join(filters)


class DashVectorStore(BasePydanticVectorStore):
    """
    Dash Vector Store.

    In this vector store, embeddings and docs are stored within a
    DashVector collection.

    During query time, the index uses DashVector to query for the top
    k most similar nodes.

    Args:
        collection (Optional[dashvector.Collection]): DashVector collection instance
        support_sparse_vector (bool): whether support sparse vector for collection.
        encoder (Optional[dashtext.SparseVectorEncoder]): encoder for generating sparse vector from document

    Examples:
        `pip install llama-index-vector-stores-dashvector`

        ```python
        import dashvector

        api_key = os.environ["DASHVECTOR_API_KEY"]
        client = dashvector.Client(api_key=api_key)

        # dimensions are for text-embedding-ada-002
        client.create("llama-demo", dimension=1536)

        dashvector_collection = client.get("quickstart")

        vector_store = DashVectorStore(dashvector_collection)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    _support_sparse_vector: bool = PrivateAttr()
    _encoder: Optional[Any] = PrivateAttr()
    _collection: Optional[Any] = PrivateAttr()

    def __init__(
        self,
        collection: Optional[Any] = None,
        support_sparse_vector: bool = False,
        encoder: Optional[Any] = None,
    ) -> None:
        """Initialize params."""
        super().__init__()

        try:
            import dashvector
        except ImportError:
            raise ImportError(
                "`dashvector` package not found, please run `pip install dashvector`"
            )

        if support_sparse_vector:
            try:
                import dashtext
            except ImportError:
                raise ImportError(
                    "`dashtext` package not found, please run `pip install dashtext`"
                )

            if encoder is None:
                encoder = dashtext.SparseVectorEncoder.default()

            self._support_sparse_vector = support_sparse_vector
            self._encoder = cast(dashtext.SparseVectorEncoder, encoder)

        if collection is not None:
            self._collection = cast(dashvector.Collection, collection)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "DashVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to vector store.

        Args:
            nodes (List[BaseNode]): list of nodes with embeddings

        """
        for i in range(0, len(nodes), DEFAULT_BATCH_SIZE):
            # batch end
            end = min(i + DEFAULT_BATCH_SIZE, len(nodes))
            docs = [
                Doc(
                    id=node.node_id,
                    vector=node.embedding,
                    sparse_vector=(
                        self._encoder.encode_documents(
                            node.get_content(metadata_mode=MetadataMode.EMBED)
                        )
                        if self._support_sparse_vector
                        else None
                    ),
                    fields=node_to_metadata_dict(
                        node, remove_text=False, flat_metadata=self.flat_metadata
                    ),
                )
                for node in nodes[i:end]
            ]

            resp = self._collection.upsert(docs)
            if not resp:
                raise Exception(f"Failed to upsert docs, error: {resp}")

        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        filter = f"{DEFAULT_DOC_ID_KEY}='{ref_doc_id}'"
        resp = self._collection.query(filter=filter)
        if not resp:
            raise Exception(f"Failed to query doc by {filter}")

        self._collection.delete(ids=[doc.id for doc in resp])

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        query_embedding = (
            [float(e) for e in query.query_embedding] if query.query_embedding else []
        )

        sparse_vector = None
        topk = query.similarity_top_k
        if (
            query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            and self._support_sparse_vector
        ):
            sparse_vector = self._encoder.encode_queries(query.query_str)
            topk = query.hybrid_top_k or query.similarity_top_k

            if query.alpha is not None:
                from dashtext import combine_dense_and_sparse

                query_embedding, sparse_vector = combine_dense_and_sparse(
                    query_embedding, sparse_vector, query.alpha
                )

        filter = _to_dashvector_filter(query.filters)
        rsp = self._collection.query(
            vector=query_embedding,
            sparse_vector=sparse_vector,
            topk=topk,
            filter=filter,
            include_vector=True,
        )
        if not rsp:
            raise Exception(f"Failed to query docs, error: {rsp}")

        top_k_ids = []
        top_k_nodes = []
        top_k_scores = []
        for doc in rsp:
            try:
                node = metadata_dict_to_node(doc.fields)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                logger.debug("Failed to parse Node metadata, fallback to legacy logic.")
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    doc.fields
                )

                text = doc.fields[DEFAULT_TEXT_KEY]
                node = TextNode(
                    id_=doc.id,
                    text=text,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            top_k_ids.append(doc.id)
            top_k_nodes.append(node)
            top_k_scores.append(doc.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
