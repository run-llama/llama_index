"""
Couchbase Vector store interface.
"""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


def _transform_couchbase_filter_condition(condition: str) -> str:
    """
    Convert standard metadata filter condition to Couchbase specific condition.

    Args:
        condition: standard metadata filter condition

    Returns:
        Couchbase specific condition
    """
    if condition == "and":
        return "conjuncts"
    elif condition == "or":
        return "disjuncts"
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_couchbase_filter_operator(
    operator: str, field: str, value: Any
) -> Dict[str, Any]:
    """
    Convert standard metadata filter operator to Couchbase specific filter operation.

    Args:
        operator: standard metadata filter operator
        field: metadata field
        value: value to apply for the filter

    Returns:
        Dictionary with Couchbase specific search operation.
    """
    if operator == "!=":
        return {"must_not": {"disjuncts": [{"field": field, "match": value}]}}
    elif operator == "==":
        return {"field": field, "match": value}
    elif operator == ">":
        return {"min": value, "inclusive_min": False, "field": field}
    elif operator == "<":
        return {"max": value, "inclusive_max": False, "field": field}
    elif operator == ">=":
        return {"min": value, "inclusive_min": True, "field": field}
    elif operator == "<=":
        return {"max": value, "inclusive_max": True, "field": field}
    elif operator == "text_match":
        return {"match_phrase": value, "field": field}
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_couchbase_filter(standard_filters: MetadataFilters) -> Dict[str, Any]:
    """
    Convert standard filters to Couchbase filter.

    Args:
        standard_filters (str): Standard Llama-index filters.

    Returns:
        Dictionary with Couchbase search query.
    """
    filters = {}
    filters_list = []
    condition = standard_filters.condition
    condition = _transform_couchbase_filter_condition(condition)

    if standard_filters.filters:
        for filter in standard_filters.filters:
            if filter.operator:
                transformed_filter = _transform_couchbase_filter_operator(
                    filter.operator, f"metadata.{filter.key}", filter.value
                )

                filters_list.append(transformed_filter)
            else:
                filters_list.append(
                    {
                        "match": {
                            "field": f"metadata.{filter.key}",
                            "value": filter.value,
                        }
                    }
                )
    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]
    elif len(filters_list) > 1:
        filters[condition] = filters_list
    return {"query": filters}


class CouchbaseVectorStore(BasePydanticVectorStore):
    """
    Couchbase Vector Store.

    To use, you should have the ``couchbase`` python package installed.

    """

    stores_text: bool = True
    flat_metadata: bool = True
    # Default batch size
    DEFAULT_BATCH_SIZE: int = 100

    _cluster: Any = PrivateAttr()
    _bucket: Any = PrivateAttr()
    _scope: Any = PrivateAttr()
    _collection: Any = PrivateAttr()
    _bucket_name: str = PrivateAttr()
    _scope_name: str = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _index_name: str = PrivateAttr()
    _id_key: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _embedding_key: str = PrivateAttr()
    _metadata_key: str = PrivateAttr()
    _scoped_index: bool = PrivateAttr()

    def __init__(
        self,
        cluster: Any,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        index_name: str,
        text_key: Optional[str] = "text",
        embedding_key: Optional[str] = "embedding",
        metadata_key: Optional[str] = "metadata",
        scoped_index: bool = True,
    ) -> None:
        """
        Initializes a connection to a Couchbase Vector Store.

        Args:
            cluster (Cluster): Couchbase cluster object with active connection.
            bucket_name (str): Name of bucket to store documents in.
            scope_name (str): Name of scope in the bucket to store documents in.
            collection_name (str): Name of collection in the scope to store documents in.
            index_name (str): Name of the Search index.
            text_key (Optional[str], optional): The field for the document text.
                Defaults to "text".
            embedding_key (Optional[str], optional): The field for the document embedding.
                Defaults to "embedding".
            metadata_key (Optional[str], optional): The field for the document metadata.
                Defaults to "metadata".
            scoped_index (Optional[bool]): specify whether the index is a scoped index.
                Set to True by default.

        Returns:
            None
        """
        try:
            from couchbase.cluster import Cluster
        except ImportError as e:
            raise ImportError(
                "Could not import couchbase python package. "
                "Please install couchbase SDK  with `pip install couchbase`."
            )

        if not isinstance(cluster, Cluster):
            raise ValueError(
                f"cluster should be an instance of couchbase.Cluster, "
                f"got {type(cluster)}"
            )

        super().__init__()
        self._cluster = cluster

        if not bucket_name:
            raise ValueError("bucket_name must be provided.")

        if not scope_name:
            raise ValueError("scope_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        if not index_name:
            raise ValueError("index_name must be provided.")

        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._index_name = index_name
        self._metadata_key = metadata_key
        self._scoped_index = scoped_index

        # Check if the bucket exists
        if not self._check_bucket_exists():
            raise ValueError(
                f"Bucket {self._bucket_name} does not exist. "
                " Please create the bucket before searching."
            )

        try:
            self._bucket = self._cluster.bucket(self._bucket_name)
            self._scope = self._bucket.scope(self._scope_name)
            self._collection = self._scope.collection(self._collection_name)
        except Exception as e:
            raise ValueError(
                "Error connecting to couchbase. "
                "Please check the connection and credentials."
            ) from e

        # Check if the scope and collection exists. Throws ValueError if they don't
        try:
            self._check_scope_and_collection_exists()
        except Exception as e:
            raise

        # Check if the index exists. Throws ValueError if it doesn't
        try:
            self._check_index_exists()
        except Exception as e:
            raise

        self._bucket = self._cluster.bucket(self._bucket_name)
        self._scope = self._bucket.scope(self._scope_name)
        self._collection = self._scope.collection(self._collection_name)

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Add nodes to the collection and return their document IDs.

        Args:
            nodes (List[BaseNode]): List of nodes to add.
            **kwargs (Any): Additional keyword arguments.
                batch_size (int): Size of the batch for batch insert.

        Returns:
            List[str]: List of document IDs for the added nodes.
        """
        from couchbase.exceptions import DocumentExistsException

        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        documents_to_insert = []
        doc_ids = []

        for node in nodes:
            metadata = node_to_metadata_dict(
                node,
                remove_text=True,
                text_field=self._text_key,
                flat_metadata=self.flat_metadata,
            )
            doc_id: str = node.node_id

            doc = {
                self._text_key: node.get_content(metadata_mode=MetadataMode.NONE),
                self._embedding_key: node.embedding,
                self._metadata_key: metadata,
            }

            documents_to_insert.append({doc_id: doc})

        for i in range(0, len(documents_to_insert), batch_size):
            batch = documents_to_insert[i : i + batch_size]
            try:
                # convert the list of dicts to a single dict for batch insert
                insert_batch = {}
                for doc in batch:
                    insert_batch.update(doc)

                logger.debug("Inserting batch of documents to Couchbase", insert_batch)

                # upsert the batch of documents into the collection
                result = self._collection.upsert_multi(insert_batch)

                logger.debug(f"Insert result: {result.all_ok}")
                if result.all_ok:
                    doc_ids.extend(insert_batch.keys())

            except DocumentExistsException as e:
                logger.debug(f"Document already exists: {e}")

            logger.debug("Inserted batch of documents to Couchbase")
        return doc_ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """
        Delete a document by its reference document ID.

        Args:
            ref_doc_id: The reference document ID to be deleted.

        Returns:
            None
        """
        try:
            document_field = self._metadata_key + ".ref_doc_id"
            query = f"DELETE FROM `{self._collection_name}` WHERE {document_field} = $ref_doc_id"
            self._scope.query(query, ref_doc_id=ref_doc_id).execute()
            logger.debug(f"Deleted document {ref_doc_id}")
        except Exception:
            logger.error(f"Error deleting document {ref_doc_id}")
            raise

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Executes a query in the vector store and returns the result.

        Args:
            query (VectorStoreQuery): The query object containing the search parameters.
            **kwargs (Any): Additional keyword arguments.
                cb_search_options (Dict): Search options to pass to Couchbase Search

        Returns:
            VectorStoreQueryResult: The result of the query containing the top-k nodes, similarities, and ids.
        """
        import couchbase.search as search
        from couchbase.options import SearchOptions
        from couchbase.vector_search import VectorQuery, VectorSearch

        fields = query.output_fields

        if not fields:
            fields = ["*"]

        # Document text field needs to be returned from the search
        if self._text_key not in fields and fields != ["*"]:
            fields.append(self._text_key)

        logger.debug("Output Fields: ", fields)

        k = query.similarity_top_k

        # Get the search options
        search_options = kwargs.get("cb_search_options", {})

        if search_options and query.filters:
            raise ValueError("Cannot use both filters and cb_search_options")
        elif query.filters:
            couchbase_options = _to_couchbase_filter(query.filters)
            logger.debug(f"Filters transformed to Couchbase: {couchbase_options}")
            search_options = couchbase_options

        logger.debug(f"Filters: {search_options}")

        # Create Search Request
        search_req = search.SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    self._embedding_key,
                    query.query_embedding,
                    k,
                )
            )
        )

        try:
            logger.debug("Querying Couchbase")
            if self._scoped_index:
                search_iter = self._scope.search(
                    self._index_name,
                    search_req,
                    SearchOptions(limit=k, fields=fields, raw=search_options),
                )

            else:
                search_iter = self._cluster.search(
                    self._index_name,
                    search_req,
                    SearchOptions(limit=k, fields=fields, raw=search_options),
                )
        except Exception as e:
            logger.debug(f"Search failed with error {e}")
            raise ValueError(f"Search failed with error: {e}")

        top_k_nodes = []
        top_k_scores = []
        top_k_ids = []

        # Parse the results
        for result in search_iter.rows():
            text = result.fields.pop(self._text_key, "")

            score = result.score

            # Format the metadata into a dictionary
            metadata_dict = self._format_metadata(result.fields)

            id = result.id

            try:
                node = metadata_dict_to_node(metadata_dict, text)
            except Exception:
                # Deprecated legacy logic for backwards compatibility
                node = TextNode(
                    text=text,
                    id_=id,
                    score=score,
                    metadata=metadata_dict,
                )

            top_k_nodes.append(node)
            top_k_scores.append(score)
            top_k_ids.append(id)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @property
    def client(self) -> Any:
        """
        Property function to access the client attribute.
        """
        return self._cluster

    def _check_bucket_exists(self) -> bool:
        """
        Check if the bucket exists in the linked Couchbase cluster.

        Returns:
            True if the bucket exists
        """
        bucket_manager = self._cluster.buckets()
        try:
            bucket_manager.get_bucket(self._bucket_name)
            return True
        except Exception as e:
            logger.debug("Error checking if bucket exists:", e)
            return False

    def _check_scope_and_collection_exists(self) -> bool:
        """
        Check if the scope and collection exists in the linked Couchbase bucket
        Returns:
            True if the scope and collection exist in the bucket
            Raises a ValueError if either is not found.
        """
        scope_collection_map: Dict[str, Any] = {}

        # Get a list of all scopes in the bucket
        for scope in self._bucket.collections().get_all_scopes():
            scope_collection_map[scope.name] = []

            # Get a list of all the collections in the scope
            for collection in scope.collections:
                scope_collection_map[scope.name].append(collection.name)

        # Check if the scope exists
        if self._scope_name not in scope_collection_map:
            raise ValueError(
                f"Scope {self._scope_name} not found in Couchbase "
                f"bucket {self._bucket_name}"
            )

        # Check if the collection exists in the scope
        if self._collection_name not in scope_collection_map[self._scope_name]:
            raise ValueError(
                f"Collection {self._collection_name} not found in scope "
                f"{self._scope_name} in Couchbase bucket {self._bucket_name}"
            )

        return True

    def _check_index_exists(self) -> bool:
        """
        Check if the Search index exists in the linked Couchbase cluster
        Returns:
            bool: True if the index exists, False otherwise.
            Raises a ValueError if the index does not exist.
        """
        if self._scoped_index:
            all_indexes = [
                index.name for index in self._scope.search_indexes().get_all_indexes()
            ]
            if self._index_name not in all_indexes:
                raise ValueError(
                    f"Index {self._index_name} does not exist. "
                    " Please create the index before searching."
                )
        else:
            all_indexes = [
                index.name for index in self._cluster.search_indexes().get_all_indexes()
            ]
            if self._index_name not in all_indexes:
                raise ValueError(
                    f"Index {self._index_name} does not exist. "
                    " Please create the index before searching."
                )

        return True

    def _format_metadata(self, row_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper method to format the metadata from the Couchbase Search API.

        Args:
            row_fields (Dict[str, Any]): The fields to format.

        Returns:
            Dict[str, Any]: The formatted metadata.
        """
        metadata = {}
        for key, value in row_fields.items():
            # Couchbase Search returns the metadata key with a prefix
            # `metadata.` We remove it to get the original metadata key
            if key.startswith(self._metadata_key):
                new_key = key.split(self._metadata_key + ".")[-1]
                metadata[new_key] = value
            else:
                metadata[key] = value

        return metadata
