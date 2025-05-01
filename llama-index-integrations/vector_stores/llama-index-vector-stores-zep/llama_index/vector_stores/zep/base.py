import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import zep_python
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from zep_python import ZepClient
from zep_python.document import Document as ZepDocument
from zep_python.document import DocumentCollection

logger = logging.getLogger(__name__)


class ZepVectorStore(BasePydanticVectorStore):
    """
    Zep Vector Store for storing and retrieving embeddings.

    Zep supports both normalized and non-normalized embeddings. Cosine similarity is
    used to compute distance and the returned score is normalized to be between 0 and 1.

    Args:
        collection_name (str): Name of the Zep collection in which to store embeddings.
        api_url (str): URL of the Zep API.
        api_key (str, optional): Key for the Zep API. Defaults to None.
        collection_description (str, optional): Description of the collection.
            Defaults to None.
        collection_metadata (dict, optional): Metadata of the collection.
            Defaults to None.
        embedding_dimensions (int, optional): Dimensions of the embeddings.
            Defaults to None.
        is_auto_embedded (bool, optional): Whether the embeddings are auto-embedded.
            Defaults to False.

    Examples:
        `pip install llama-index-vector-stores-zep`

        ```python
        from llama_index.vector_stores.zep import ZepVectorStore

        vector_store = ZepVectorStore(
            api_url="<api_url>",
            api_key="<api_key>",
            collection_name="<unique_collection_name>",  # Can either be an existing collection or a new one
            embedding_dimensions=1536,  # Optional, required if creating a new collection
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    _client: ZepClient = PrivateAttr()
    _collection: DocumentCollection = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        api_url: str,
        api_key: Optional[str] = None,
        collection_description: Optional[str] = None,
        collection_metadata: Optional[Dict[str, Any]] = None,
        embedding_dimensions: Optional[int] = None,
        is_auto_embedded: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__()

        self._client = ZepClient(base_url=api_url, api_key=api_key)
        collection: Union[DocumentCollection, None] = None

        try:
            collection = self._client.document.get_collection(name=collection_name)
        except zep_python.NotFoundError:
            if embedding_dimensions is None:
                raise ValueError(
                    "embedding_dimensions must be specified if collection does not"
                    " exist"
                )
            logger.info(
                f"Collection {collection_name} does not exist, "
                f"will try creating one with dimensions={embedding_dimensions}"
            )

            collection = self._client.document.add_collection(
                name=collection_name,
                embedding_dimensions=embedding_dimensions,
                is_auto_embedded=is_auto_embedded,
                description=collection_description,
                metadata=collection_metadata,
            )

        assert collection is not None
        self._collection = collection

    @classmethod
    def class_name(cls) -> str:
        return "ZepVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def _prepare_documents(
        self, nodes: List[BaseNode]
    ) -> Tuple[List["ZepDocument"], List[str]]:
        docs: List["ZepDocument"] = []
        ids: List[str] = []

        for node in nodes:
            metadata_dict: Dict[str, Any] = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )

            if len(node.get_content()) == 0:
                raise ValueError("No content to add to Zep")

            docs.append(
                ZepDocument(
                    document_id=node.node_id,
                    content=node.get_content(metadata_mode=MetadataMode.NONE),
                    embedding=node.get_embedding(),
                    metadata=metadata_dict,
                )
            )
            ids.append(node.node_id)

        return docs, ids

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the collection.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings.

        Returns:
            List[str]: List of IDs of the added documents.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if self._collection.is_auto_embedded:
            raise ValueError("Collection is auto embedded, cannot add embeddings")

        docs, ids = self._prepare_documents(nodes)

        self._collection.add_documents(docs)

        return ids

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes to the collection.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings.

        Returns:
            List[str]: List of IDs of the added documents.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if self._collection.is_auto_embedded:
            raise ValueError("Collection is auto embedded, cannot add embeddings")

        docs, ids = self._prepare_documents(nodes)

        await self._collection.aadd_documents(docs)

        return ids

    def delete(
        self, ref_doc_id: Optional[str] = None, **delete_kwargs: Any
    ) -> None:  # type: ignore
        """
        Delete a document from the collection.

        Args:
            ref_doc_id (Optional[str]): ID of the document to delete.
                Not currently supported.
            delete_kwargs: Must contain "uuid" key with UUID of the document to delete.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if ref_doc_id and len(ref_doc_id) > 0:
            raise NotImplementedError(
                "Delete by ref_doc_id not yet implemented for Zep."
            )

        if "uuid" in delete_kwargs:
            self._collection.delete_document(uuid=delete_kwargs["uuid"])
        else:
            raise ValueError("uuid must be specified")

    async def adelete(
        self, ref_doc_id: Optional[str] = None, **delete_kwargs: Any
    ) -> None:  # type: ignore
        """
        Asynchronously delete a document from the collection.

        Args:
            ref_doc_id (Optional[str]): ID of the document to delete.
                Not currently supported.
            delete_kwargs: Must contain "uuid" key with UUID of the document to delete.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if ref_doc_id and len(ref_doc_id) > 0:
            raise NotImplementedError(
                "Delete by ref_doc_id not yet implemented for Zep."
            )

        if "uuid" in delete_kwargs:
            await self._collection.adelete_document(uuid=delete_kwargs["uuid"])
        else:
            raise ValueError("uuid must be specified")

    def _parse_query_result(
        self, results: List["ZepDocument"]
    ) -> VectorStoreQueryResult:
        similarities: List[float] = []
        ids: List[str] = []
        nodes: List[TextNode] = []

        for d in results:
            node = metadata_dict_to_node(d.metadata or {})
            node.set_content(d.content)

            nodes.append(node)

            if d.score is None:
                d.score = 0.0
            similarities.append(d.score)

            if d.document_id is None:
                d.document_id = ""
            ids.append(d.document_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _to_zep_filters(self, filters: MetadataFilters) -> Dict[str, Any]:
        """Convert filters to Zep filters. Filters are ANDed together."""
        filter_conditions: List[Dict[str, Any]] = []

        for f in filters.legacy_filters():
            filter_conditions.append({"jsonpath": f'$[*] ? (@.{f.key} == "{f.value}")'})

        return {"where": {"and": filter_conditions}}

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query the index for the top k most similar nodes to the given query.

        Args:
            query (VectorStoreQuery): Query object containing either a query string
                or a query embedding.

        Returns:
            VectorStoreQueryResult: Result of the query, containing the most similar
                nodes, their similarities, and their IDs.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if query.query_embedding is None and query.query_str is None:
            raise ValueError("query must have one of query_str or query_embedding")

        # If we have an embedding, we shouldn't use the query string
        # Zep does not allow both to be set
        if query.query_embedding:
            query.query_str = None

        metadata_filters = None
        if query.filters is not None:
            metadata_filters = self._to_zep_filters(query.filters)

        results = self._collection.search(
            text=query.query_str,
            embedding=query.query_embedding,
            metadata=metadata_filters,
            limit=query.similarity_top_k,
        )

        return self._parse_query_result(results)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query the index for the top k most similar nodes to the
            given query.

        Args:
            query (VectorStoreQuery): Query object containing either a query string or
                a query embedding.

        Returns:
            VectorStoreQueryResult: Result of the query, containing the most similar
                nodes, their similarities, and their IDs.

        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError("Collection not initialized")

        if query.query_embedding is None and query.query_str is None:
            raise ValueError("query must have one of query_str or query_embedding")

        # If we have an embedding, we shouldn't use the query string
        # Zep does not allow both to be set
        if query.query_embedding:
            query.query_str = None

        metadata_filters = None
        if query.filters is not None:
            metadata_filters = self._to_zep_filters(query.filters)

        results = await self._collection.asearch(
            text=query.query_str,
            embedding=query.query_embedding,
            metadata=metadata_filters,
            limit=query.similarity_top_k,
        )

        return self._parse_query_result(results)
