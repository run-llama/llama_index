"""Azure Cognitive Search vector store."""
import enum
import json
import logging
from enum import auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class MetadataIndexFieldType(int, enum.Enum):
    """
    Enumeration representing the supported types for metadata fields in an
    Azure Cognitive Search Index, corresponds with types supported in a flat
    metadata dictionary.
    """

    STRING = auto()  # "Edm.String"
    BOOLEAN = auto()  # "Edm.Boolean"
    INT32 = auto()  # "Edm.Int32"
    INT64 = auto()  # "Edm.Int64"
    DOUBLE = auto()  # "Edm.Double"


class IndexManagement(int, enum.Enum):
    """Enumeration representing the supported index management operations."""

    NO_VALIDATION = auto()
    VALIDATE_INDEX = auto()
    CREATE_IF_NOT_EXISTS = auto()


class CognitiveSearchVectorStore(VectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    def _normalise_metadata_to_index_fields(
        self,
        filterable_metadata_field_keys: Union[
            List[str],
            Dict[str, str],
            Dict[str, Tuple[str, MetadataIndexFieldType]],
            None,
        ] = [],
    ) -> Dict[str, Tuple[str, MetadataIndexFieldType]]:
        index_field_spec: Dict[str, Tuple[str, MetadataIndexFieldType]] = {}

        if isinstance(filterable_metadata_field_keys, List):
            for field in filterable_metadata_field_keys:
                # Index field name and the metadata field name are the same
                # Use String as the default index field type
                index_field_spec[field] = (field, MetadataIndexFieldType.STRING)

        elif isinstance(filterable_metadata_field_keys, Dict):
            for k, v in filterable_metadata_field_keys.items():
                if isinstance(v, tuple):
                    # Index field name and metadata field name may differ
                    # The index field type used is as supplied
                    index_field_spec[k] = v
                else:
                    # Index field name and metadata field name may differ
                    # Use String as the default index field type
                    index_field_spec[k] = (v, MetadataIndexFieldType.STRING)

        return index_field_spec

    def _create_index_if_not_exists(self, index_name: str) -> None:
        if index_name not in self._index_client.list_index_names():
            logger.info(f"Index {index_name} does not exist, creating index")
            self._create_index(index_name)

    def _create_metadata_index_fields(self) -> List[Any]:
        """Create a list of index fields for storing metadata values."""
        from azure.search.documents.indexes.models import SimpleField

        index_fields = []

        # create search fields
        for v in self._metadata_to_index_field_map.values():
            field_name, field_type = v

            if field_type == MetadataIndexFieldType.STRING:
                index_field_type = "Edm.String"
            elif field_type == MetadataIndexFieldType.INT32:
                index_field_type = "Edm.Int32"
            elif field_type == MetadataIndexFieldType.INT64:
                index_field_type = "Edm.Int64"
            elif field_type == MetadataIndexFieldType.DOUBLE:
                index_field_type = "Edm.Double"
            elif field_type == MetadataIndexFieldType.BOOLEAN:
                index_field_type = "Edm.Boolean"

            field = SimpleField(name=field_name, type=index_field_type, filterable=True)
            index_fields.append(field)

        return index_fields

    def _create_index(self, index_name: Optional[str]) -> None:
        """
        Creates a default index based on the supplied index name, key field names and
        metadata filtering keys.
        """
        from azure.search.documents.indexes.models import (
            HnswParameters,
            HnswVectorSearchAlgorithmConfiguration,
            PrioritizedFields,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticSettings,
            SimpleField,
            VectorSearch,
        )

        logger.info(f"Configuring {index_name} fields")
        fields = [
            SimpleField(name=self._field_mapping["id"], type="Edm.String", key=True),
            SearchableField(
                name=self._field_mapping["chunk"],
                type="Edm.String",
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name=self._field_mapping["embedding"],
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
                vector_search_dimensions=self.embedding_dimensionality,
                vector_search_configuration="default",
            ),
            SimpleField(name=self._field_mapping["metadata"], type="Edm.String"),
            SimpleField(
                name=self._field_mapping["doc_id"], type="Edm.String", filterable=True
            ),
        ]

        logger.info(f"Configuring {index_name} metadata fields")
        # Add on the metadata fields

        metadata_index_fields = self._create_metadata_index_fields()

        fields.extend(metadata_index_fields)

        logger.info(f"Configuring {index_name} vector search")

        hnsw_param = HnswParameters(
            m=4,
            ef_construction=500,
            ef_search=1000,
            metric="cosine",
        )

        vector_search = VectorSearch(
            algorithm_configurations=[
                HnswVectorSearchAlgorithmConfiguration(
                    name="default",
                    kind="hnsw",
                    parameters=hnsw_param,
                )
            ]
        )

        logger.info(f"Configuring {index_name} semantic search")
        semantic_settings = SemanticSettings(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=PrioritizedFields(
                        title_field=None,
                        prioritized_content_fields=[
                            SemanticField(field_name=self._field_mapping["chunk"])
                        ],
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            semantic_settings=semantic_settings,
            vector_search=vector_search,
        )

        logger.debug(f"Creating {index_name} search index")
        self._index_client.create_index(index)

    def _validate_index(self, index_name: Optional[str]) -> None:
        if self._index_client and index_name:
            if index_name not in self._index_client.list_index_names():
                raise ValueError(
                    f"Validation failed, index {index_name} does not exist."
                )

    def __init__(
        self,
        search_or_index_client: Any,
        id_field_key: str,
        chunk_field_key: str,
        embedding_field_key: str,
        metadata_string_field_key: str,
        doc_id_field_key: str,
        filterable_metadata_field_keys: Optional[
            Union[
                List[str],
                Dict[str, str],
                Dict[str, Tuple[str, MetadataIndexFieldType]],
            ]
        ] = None,
        index_name: Optional[str] = None,
        index_mapping: Optional[
            Callable[[Dict[str, str], Dict[str, Any]], Dict[str, str]]
        ] = None,
        index_management: IndexManagement = IndexManagement.NO_VALIDATION,
        embedding_dimensionality: int = 1536,
        **kwargs: Any,
    ) -> None:
        # ruff: noqa: E501
        """
        Embeddings and documents are stored in an Azure Cognitive Search index,
        a merge or upload approach is used when adding embeddings.
        When adding multiple embeddings the index is updated by this vector store
        in batches of 10 documents, very large nodes may result in failure due to
        the batch byte size being exceeded.

        Args:
            search_client (azure.search.documents.SearchClient):
                Client for index to populated / queried.
            id_field_key (str): Index field storing the id
            chunk_field_key (str): Index field storing the node text
            embedding_field_key (str): Index field storing the embedding vector
            metadata_string_field_key (str):
                Index field storing node metadata as a json string.
                Schema is arbitrary, to filter on metadata values they must be stored
                as separate fields in the index, use filterable_metadata_field_keys
                to specify the metadata values that should be stored in these filterable fields
            doc_id_field_key (str): Index field storing doc_id
            index_mapping:
                Optional function with definition
                (enriched_doc: Dict[str, str], metadata: Dict[str, Any]): Dict[str,str]
                used to map document fields to the Cognitive search index fields
                (return value of function).
                If none is specified a default mapping is provided which uses
                the field keys. The keys in the enriched_doc are
                ["id", "chunk", "embedding", "metadata"]
                The default mapping is:
                    - "id" to id_field_key
                    - "chunk" to chunk_field_key
                    - "embedding" to embedding_field_key
                    - "metadata" to metadata_field_key
            *kwargs (Any): Additional keyword arguments.

        Raises:
            ImportError: Unable to import `azure-search-documents`
            ValueError: If `search_or_index_client` is not provided
            ValueError: If `index_name` is not provided and `search_or_index_client`
                is of type azure.search.documents.SearchIndexClient
            ValueError: If `index_name` is provided and `search_or_index_client`
                is of type azure.search.documents.SearchClient
            ValueError: If `create_index_if_not_exists` is true and
                `search_or_index_client` is of type azure.search.documents.SearchClient
        """
        import_err_msg = (
            "`azure-search-documents` package not found, please run "
            "`pip install azure-search-documents==11.4.0b8`"
        )

        try:
            import azure.search.documents  # noqa
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
        except ImportError:
            raise ImportError(import_err_msg)

        self._index_client: SearchIndexClient = cast(SearchIndexClient, None)
        self._search_client: SearchClient = cast(SearchClient, None)
        self.embedding_dimensionality = embedding_dimensionality

        # Validate search_or_index_client
        if search_or_index_client is not None:
            if isinstance(search_or_index_client, SearchIndexClient):
                # If SearchIndexClient is supplied so must index_name
                self._index_client = cast(SearchIndexClient, search_or_index_client)

                if not index_name:
                    raise ValueError(
                        "index_name must be supplied if search_or_index_client is of "
                        "type azure.search.documents.SearchIndexClient"
                    )

                self._search_client = self._index_client.get_search_client(
                    index_name=index_name
                )

            elif isinstance(search_or_index_client, SearchClient):
                self._search_client = cast(SearchClient, search_or_index_client)

                # Validate index_name
                if index_name:
                    raise ValueError(
                        "index_name cannot be supplied if search_or_index_client "
                        "is of type azure.search.documents.SearchClient"
                    )

            if not self._index_client and not self._search_client:
                raise ValueError(
                    "search_or_index_client must be of type "
                    "azure.search.documents.SearchClient or "
                    "azure.search.documents.SearchIndexClient"
                )
        else:
            raise ValueError("search_or_index_client not specified")

        if (
            index_management == IndexManagement.CREATE_IF_NOT_EXISTS
            and not self._index_client
        ):
            raise ValueError(
                "index_management has value of IndexManagement.CREATE_IF_NOT_EXISTS "
                "but search_or_index_client is not of type "
                "azure.search.documents.SearchIndexClient"
            )

        self._index_management = index_management

        # Default field mapping
        field_mapping = {
            "id": id_field_key,
            "chunk": chunk_field_key,
            "embedding": embedding_field_key,
            "metadata": metadata_string_field_key,
            "doc_id": doc_id_field_key,
        }

        self._field_mapping = field_mapping

        self._index_mapping = (
            self._default_index_mapping if index_mapping is None else index_mapping
        )

        # self._filterable_metadata_field_keys = filterable_metadata_field_keys
        self._metadata_to_index_field_map = self._normalise_metadata_to_index_fields(
            filterable_metadata_field_keys
        )

        if self._index_management == IndexManagement.CREATE_IF_NOT_EXISTS:
            if index_name:
                self._create_index_if_not_exists(index_name)

        if self._index_management == IndexManagement.VALIDATE_INDEX:
            self._validate_index(index_name)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._search_client

    def _default_index_mapping(
        self, enriched_doc: Dict[str, str], metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        index_doc: Dict[str, str] = {}

        for field in self._field_mapping:
            index_doc[self._field_mapping[field]] = enriched_doc[field]

        for metadata_field_name, (
            index_field_name,
            _,
        ) in self._metadata_to_index_field_map.items():
            metadata_value = metadata.get(metadata_field_name)
            if metadata_value:
                index_doc[index_field_name] = metadata_value

        return index_doc

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index associated with the configured search client.

        Args:
            nodes: List[BaseNode]: nodes with embeddings

        """
        if not self._search_client:
            raise ValueError("Search client not initialized")

        documents = []
        ids = []

        for node in nodes:
            logger.debug(f"Processing embedding: {node.node_id}")
            ids.append(node.node_id)

            index_document = self._create_index_document(node)

            documents.append(index_document)

            if len(documents) >= 10:
                logger.info(
                    f"Uploading batch of size {len(documents)}, "
                    f"current progress {len(ids)} of {len(nodes)}"
                )
                self._search_client.merge_or_upload_documents(documents)
                documents = []

        # Upload remaining batch of less than 10 documents
        if len(documents) > 0:
            logger.info(
                f"Uploading remaining batch of size {len(documents)}, "
                f"current progress {len(ids)} of {len(nodes)}"
            )
            self._search_client.merge_or_upload_documents(documents)
            documents = []

        return ids

    def _create_index_document(self, node: BaseNode) -> Dict[str, Any]:
        """Create Cognitive Search index document from embedding result."""
        doc: Dict[str, Any] = {}
        doc["id"] = node.node_id
        doc["chunk"] = node.get_content(metadata_mode=MetadataMode.NONE) or ""
        doc["embedding"] = node.get_embedding()
        doc["doc_id"] = node.ref_doc_id

        node_metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )

        doc["metadata"] = json.dumps(node_metadata)

        return self._index_mapping(doc, node_metadata)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete documents from the Cognitive Search Index
        with doc_id_field_key field equal to ref_doc_id.
        """
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

    def _create_odata_filter(self, metadata_filters: MetadataFilters) -> str:
        """Generate an OData filter string using supplied metadata filters."""
        odata_filter: List[str] = []
        for f in metadata_filters.filters:
            if not isinstance(f, ExactMatchFilter):
                raise NotImplementedError(
                    "Only `ExactMatchFilter` filters are supported"
                )

            # Raise error if filtering on a metadata field that lacks a mapping to
            # an index field
            metadata_mapping = self._metadata_to_index_field_map.get(f.key)

            if not metadata_mapping:
                raise ValueError(
                    f"Metadata field '{f.key}' is missing a mapping to an index field, "
                    "provide entry in 'filterable_metadata_field_keys' for this "
                    "vector store"
                )

            index_field = metadata_mapping[0]

            if len(odata_filter) > 0:
                odata_filter.append(" and ")
            if isinstance(f.value, str):
                escaped_value = "".join([("''" if s == "'" else s) for s in f.value])
                odata_filter.append(f"{index_field} eq '{escaped_value}'")
            else:
                odata_filter.append(f"{index_field} eq {f.value}")

        odata_expr = "".join(odata_filter)

        logger.info(f"Odata filter: {odata_expr}")

        return odata_expr

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        from azure.search.documents.models import Vector

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
            logger.info("Vector search with supplied embedding")

        odata_filter = None
        if query.filters is not None:
            odata_filter = self._create_odata_filter(query.filters)

        results = self._search_client.search(
            search_text=search_query,
            vectors=vectors,
            top=query.similarity_top_k,
            select=select_fields,
            filter=odata_filter,
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

        logger.debug(
            f"Search query '{search_query}' returned {len(id_result)} results."
        )

        return VectorStoreQueryResult(
            nodes=node_result, similarities=score_result, ids=id_result
        )
