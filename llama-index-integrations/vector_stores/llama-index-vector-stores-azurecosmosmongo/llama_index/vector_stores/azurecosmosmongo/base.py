"""Azure CosmosDB MongoDB vCore Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, Dict, List, Optional, cast
from datetime import date

import pymongo
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class AzureCosmosDBMongoDBVectorSearch(BasePydanticVectorStore):
    """Azure CosmosDB MongoDB vCore Vector Store.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with an Azure Cosmodb MongoDB vCore Cluster

    Examples:
        `pip install llama-index-vector-stores-azurecosmosmongo`

        ```python
        import pymongo
        from llama_index.vector_stores.azurecosmosmongo import AzureCosmosDBMongoDBVectorSearch

        # Set up the connection string with your Azure CosmosDB MongoDB URI
        connection_string = "YOUR_AZURE_COSMOSDB_MONGODB_URI"
        mongodb_client = pymongo.MongoClient(connection_string)

        # Create an instance of AzureCosmosDBMongoDBVectorSearch
        vector_store = AzureCosmosDBMongoDBVectorSearch(
            mongodb_client=mongodb_client,
            db_name="demo_vectordb",
            collection_name="paul_graham_essay",
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _collection: Any = PrivateAttr()
    _index_name: str = PrivateAttr()
    _embedding_key: str = PrivateAttr()
    _id_key: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _metadata_key: str = PrivateAttr()
    _insert_kwargs: dict = PrivateAttr()
    _db_name: str = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _cosmos_search_kwargs: dict = PrivateAttr()
    _mongodb_client: Any = PrivateAttr()

    def __init__(
        self,
        mongodb_client: Optional[Any] = None,
        db_name: str = "default_db",
        collection_name: str = "default_collection",
        index_name: str = "default_vector_search_index",
        id_key: str = "id",
        embedding_key: str = "content_vector",
        text_key: str = "text",
        metadata_key: str = "metadata",
        cosmos_search_kwargs: Optional[Dict] = None,
        insert_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vector store.

        Args:
            mongodb_client: An Azure CosmoDB MongoDB client (type: MongoClient, shown any for lazy import).
            db_name: An Azure CosmosDB MongoDB database name.
            collection_name: An Azure CosmosDB collection name.
            index_name: An Azure CosmosDB MongoDB vCore Vector Search index name.
            id_key: The data field to use as the id.
            embedding_key: An Azure CosmosDB MongoDB field that will contain
            the embedding for each document.
            text_key: An Azure CosmosDB MongoDB field that will contain the text for each document.
            metadata_key: An Azure CosmosDB MongoDB field that will contain
            the metadata for each document.
            cosmos_search_kwargs: An Azure CosmosDB MongoDB field that will
            contain search options, such as kind, numLists, similarity, and dimensions.
            insert_kwargs: The kwargs used during `insert`.
        """
        super().__init__()

        if mongodb_client is not None:
            self._mongodb_client = cast(pymongo.MongoClient, mongodb_client)
        else:
            if "AZURE_COSMOSDB_MONGODB_URI" not in os.environ:
                raise ValueError(
                    "Must specify Azure cosmodb 'AZURE_COSMOSDB_MONGODB_URI' via env variable "
                    "if not directly passing in client."
                )
            self._mongodb_client = pymongo.MongoClient(
                os.environ["AZURE_COSMOSDB_MONGODB_URI"],
                appname="LlamaIndex-CDBMongoVCore-VectorStore-Python",
            )

        self._collection = self._mongodb_client[db_name][collection_name]
        self._index_name = index_name
        self._embedding_key = embedding_key
        self._id_key = id_key
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._insert_kwargs = insert_kwargs or {}
        self._db_name = db_name
        self._collection_name = collection_name
        self._cosmos_search_kwargs = cosmos_search_kwargs or {}
        self._create_vector_search_index()

    def _create_vector_search_index(self) -> None:
        db = self._mongodb_client[self._db_name]

        create_index_commands = {}
        kind = self._cosmos_search_kwargs.get("kind", "vector-hnsw")

        if kind == "vector-ivf":
            create_index_commands = self._get_vector_index_ivf(kind)
        elif kind == "vector-hnsw":
            create_index_commands = self._get_vector_index_hnsw(kind)
        db.command(create_index_commands)

    def _get_vector_index_ivf(
        self,
        kind: str,
    ) -> Dict[str, Any]:
        return {
            "createIndexes": self._collection_name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": kind,
                        "numLists": self._cosmos_search_kwargs.get("numLists", 1),
                        "similarity": self._cosmos_search_kwargs.get(
                            "similarity", "COS"
                        ),
                        "dimensions": self._cosmos_search_kwargs.get(
                            "dimensions", 1536
                        ),
                    },
                }
            ],
        }

    def _get_vector_index_hnsw(
        self,
        kind: str,
    ) -> Dict[str, Any]:
        return {
            "createIndexes": self._collection_name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": kind,
                        "m": self._cosmos_search_kwargs.get("m", 2),
                        "efConstruction": self._cosmos_search_kwargs.get(
                            "efConstruction", 64
                        ),
                        "similarity": self._cosmos_search_kwargs.get(
                            "similarity", "COS"
                        ),
                        "dimensions": self._cosmos_search_kwargs.get(
                            "dimensions", 1536
                        ),
                    },
                }
            ],
        }

    def create_filter_index(
        self,
        property_to_filter: str,
        index_name: str,
    ) -> dict[str, Any]:
        db = self._mongodb_client[self._db_name]
        command = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "key": {property_to_filter: 1},
                    "name": index_name,
                }
            ],
        }

        create_index_responses: dict[str, Any] = db.command(command)
        return create_index_responses

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            A List of ids for successfully added nodes.

        """
        ids = []
        data_to_insert = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )

            entry = {
                self._id_key: node.node_id,
                self._embedding_key: node.get_embedding(),
                self._text_key: node.get_content(metadata_mode=MetadataMode.NONE) or "",
                self._metadata_key: metadata,
                "timeStamp": date.today(),
            }
            data_to_insert.append(entry)
            ids.append(node.node_id)
        logger.debug("Inserting data into MongoDB: %s", data_to_insert)
        insert_result = self._collection.insert_many(
            data_to_insert, **self._insert_kwargs
        )
        logger.debug("Result of insert: %s", insert_result)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # delete by filtering on the doc_id metadata
        self._collection.delete_one(
            filter={self._metadata_key + ".ref_doc_id": ref_doc_id}, **delete_kwargs
        )

    @property
    def client(self) -> Any:
        """Return MongoDB client."""
        return self._mongodb_client

    def _query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        pipeline: List[dict[str, Any]] = []
        kind = self._cosmos_search_kwargs.get("kind", "vector-hnsw")
        if kind == "vector-ivf":
            pipeline = self._get_pipeline_vector_ivf(
                query, kwargs.get("pre_filter", {})
            )
        elif kind == "vector-hnsw":
            pipeline = self._get_pipeline_vector_hnsw(
                query, kwargs.get("ef_search", 40), kwargs.get("pre_filter", {})
            )

        logger.debug("Running query pipeline: %s", pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for res in cursor:
            text = res["document"].pop(self._text_key)
            score = res.pop("similarityScore")
            id = res["document"].pop(self._id_key)
            metadata_dict = res["document"].pop(self._metadata_key)

            try:
                node = metadata_dict_to_node(metadata_dict)
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata_dict
                )

                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            top_k_ids.append(id)
            top_k_nodes.append(node)
            top_k_scores.append(score)
        result = VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
        logger.debug("Result of query: %s", result)
        return result

    def _get_pipeline_vector_ivf(
        self, query: VectorStoreQuery, pre_filter: Optional[Dict]
    ) -> List[dict[str, Any]]:
        params = {
            "vector": query.query_embedding,
            "path": self._embedding_key,
            "k": query.similarity_top_k,
        }
        if pre_filter:
            params["filter"] = pre_filter

        pipeline: List[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                    "returnStoredSource": True,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def _get_pipeline_vector_hnsw(
        self, query: VectorStoreQuery, ef_search: int, pre_filter: Optional[Dict]
    ) -> List[dict[str, Any]]:
        params = {
            "vector": query.query_embedding,
            "path": self._embedding_key,
            "k": query.similarity_top_k,
            "efSearch": ef_search,
        }
        if pre_filter:
            params["filter"] = pre_filter

        pipeline: List[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: a VectorStoreQuery object.

        Returns:
            A VectorStoreQueryResult containing the results of the query.
        """
        return self._query(query, **kwargs)
