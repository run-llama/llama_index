"""Azure Cognitive Search vector store."""
import logging
import math
from typing import Any, List, cast, Dict, Callable, Optional

from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    legacy_metadata_dict_to_node,
)


import json

logger = logging.getLogger(__name__)


class CognitiveSearchVectorStore(VectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        search_client: Any,
        id_field_key: str,
        chunk_field_key: str,
        embedding_field_key: str,
        metadata_field_key: str,
        doc_id_field_key: str,
        index_mapping: Optional[
            Callable[[Dict[str, str], Dict[str, Any]], Dict[str, str]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        Embeddings and documents are stored in an Azure Cognitive Search index, a merge or upload approach is used when adding embeddings.
        When adding multiple embeddings the index is updated by this vector store in batches of 10 documents,
        very large nodes may result in failure due to the batch byte size being exceeded.

        Args:
            search_client (azure.search.documents.SearchClient): Client for index to populated / queried.
            id_field_key (str): Index field storing the id
            chunk_field_key (str): Index field storing the node text
            embedding_field_key (str): Index field storing the embedding vector
            metadata_field_key (str): Index field storing node metadata
            doc_id_field_key (str): Index field storing doc_id
            index_mapping: Optional function with definition (enriched_doc: Dict[str, str], metadata: Dict[str, Any]): Dict[str,str]
                used to map document fields to the Cognitive search index fields (return value of function).
                If none is specified a default mapping is provided which uses the field keys
                The keys in the enriched_doc are ["id", "chunk", "embedding", "metadata"]
                The default mapping is:
                    - "id" to id_field_key
                    - "chunk" to chunk_field_key
                    - "embedding" to embedding_field_key
                    - "metadata" to metadata_field_key
            *kwargs (Any): Additional keyword arguments.

        Raises:
            ImportError: Unable to import `azure.search.documents`
            ValueError: If `search_client` is not provided
        """

        import_err_msg = "`azure-search-documents` package not found, please run `pip install azure-search-documents==11.4.0b8`"
        try:
            import azure.search.documents  # noqa: F401
            from azure.search.documents import SearchClient
        except ImportError:
            raise ImportError(import_err_msg)

        if search_client is not None:
            self._search_client = cast(SearchClient, search_client)
        else:
            raise ValueError("search_client not specified")

        # Default field mapping
        field_mapping = {
            "id": id_field_key,
            "chunk": chunk_field_key,
            "embedding": embedding_field_key,
            "metadata": metadata_field_key,
            "doc_id": doc_id_field_key,
        }

        self._field_mapping = field_mapping

        self._index_mapping = (
            self._default_index_mapping if index_mapping is None else index_mapping
        )

    @property
    def client(self) -> Any:
        """Get client."""
        return self._search_client

    def _default_index_mapping(
        self, enriched_doc: Dict[str, str], metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        index_doc: Dict[str, str] = {}

        for field in self._field_mapping.keys():
            index_doc[self._field_mapping[field]] = enriched_doc[field]

        return index_doc

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index associated with the configured search client.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """

        if not self._search_client:
            raise ValueError("Search client not initialized")

        documents = []
        ids = []

        for embedding in embedding_results:
            logger.debug(f"Processing embedding: {embedding.id}")
            ids.append(embedding.id)

            index_document = self._create_index_document(embedding)

            documents.append(index_document)

            if len(documents) >= 10:
                logger.info(
                    f"Uploading batch of size {len(documents)}, current progress {len(ids)} of {len(embedding_results)}"
                )
                self._search_client.merge_or_upload_documents(documents)
                documents = []

        # Upload remaining batch of less than 10 documents
        if len(documents) > 0:
            logger.info(
                f"Uploading remaining batch of size {len(documents)}, current progress {len(ids)} of {len(embedding_results)}"
            )
            self._search_client.merge_or_upload_documents(documents)
            documents = []

        return ids

    def _create_index_document(self, embedding: NodeWithEmbedding) -> Dict[str, Any]:
        """Create Cognitive Search index document from embedding result"""
        doc: Dict[str, Any] = {}
        doc["id"] = embedding.id
        doc["chunk"] = embedding.node.get_content(metadata_mode=MetadataMode.NONE) or ""
        doc["embedding"] = embedding.embedding
        doc["doc_id"] = embedding.ref_doc_id

        node_metadata = node_to_metadata_dict(
            embedding.node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )

        doc["metadata"] = json.dumps(node_metadata)

        index_document = self._index_mapping(doc, node_metadata)

        return index_document

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete documents from the Cognitive Search Index
        with doc_id_field_key field equal to ref_doc_id."""

        # Locate documents to delete
        filter = f'{self._field_mapping["doc_id"]} eq \'{ref_doc_id}\''
        results = self._search_client.search(search_text="*", filter=filter)

        logger.debug(f"Searching with filter {filter}")

        docs_to_delete = []
        for result in results:
            doc = {}
            doc["id"] = result[self._field_mapping["id"]]
            logger.debug(f"Found document to delete: {doc}")
            docs_to_delete.append(doc)

        if len(docs_to_delete) > 0:
            logger.debug(f"Deleting {len(docs_to_delete)} documents")
            self._search_client.delete_documents(docs_to_delete)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        from azure.search.documents.models import Vector

        if query.filters is not None:
            raise ValueError(
                "Metadata filters not implemented for CognitiveSearchVectorStore yet."
            )

        select_fields = [
            self._field_mapping["id"],
            self._field_mapping["chunk"],
            self._field_mapping["metadata"],
            self._field_mapping["doc_id"],
        ]

        search_query = "*"
        vectors = None

        if query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID):
            if query.query_str is None:
                raise ValueError("Query missing query string")

            search_query = query.query_str

            logger.info(f"Hybrid search with search text: {search_query}")

        if query.mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID):
            if not query.query_embedding:
                raise ValueError("Query missing embedding")

            vector = Vector(
                value=query.query_embedding,
                k=query.similarity_top_k,
                fields=self._field_mapping["embedding"],
            )
            vectors = [vector]
            logger.info(f"Vector search with supplied embedding")

        results = self._search_client.search(
            search_text=search_query,
            vectors=vectors,
            top=query.similarity_top_k,
            select=select_fields,
        )

        id_result = []
        node_result = []
        score_result = []
        for result in results:
            node_id = result[self._field_mapping["id"]]
            metadata = json.loads(result[self._field_mapping["metadata"]])
            score = result["@search.score"]
            chunk = result[self._field_mapping["chunk"]]

            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=chunk,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            logger.debug(f"Retrieved node id {node_id} with node data of {node}")

            id_result.append(node_id)
            node_result.append(node)
            score_result.append(score)

        return VectorStoreQueryResult(
            nodes=node_result, similarities=score_result, ids=id_result
        )
