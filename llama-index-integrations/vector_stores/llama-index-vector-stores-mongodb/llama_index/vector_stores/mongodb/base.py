"""
MongoDB Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from importlib.metadata import version
from typing import Any, Dict, List, Optional, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.mongodb.pipelines import (
    combine_pipelines,
    filters_to_mql,
    final_hybrid_stage,
    fulltext_search_stage,
    reciprocal_rank_stage,
    vector_search_stage,
)
from .index import (
    create_vector_search_index,
    drop_vector_search_index,
    update_vector_search_index,
    create_fulltext_search_index,
)
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo

logger = logging.getLogger(__name__)


class MongoDBAtlasVectorSearch(BasePydanticVectorStore):
    """
    MongoDB Atlas Vector Store.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB Atlas Cluster
    that has an Atlas Vector Search index

    To get started head over to the [Atlas quick start](https://www.mongodb.com/docs/atlas/getting-started/).

    Once your store is created, be sure to enable indexing in the Atlas GUI.

    Please refer to the [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/)
    to get more details on how to define an Atlas Vector Search index. You can name the index {ATLAS_VECTOR_SEARCH_INDEX_NAME}
    and create the index on the namespace {DB_NAME}.{COLLECTION_NAME}.

    Finally, write the following definition in the JSON editor on MongoDB Atlas:

    ```
    {
        "name": "vector_index",
        "type": "vectorSearch",
        "fields":[
            {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine"
            }
        ]
    }
    ```

    Optionally, you can use the experimental convenience methods on this class to manage the vector search
    index and the full text index.


    Examples:
        `pip install llama-index-vector-stores-mongodb`

        ```python
        import pymongo
        from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

        # Ensure you have the MongoDB URI with appropriate credentials
        mongo_uri = "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority"
        mongodb_client = pymongo.MongoClient(mongo_uri)

        # Create an instance of MongoDBAtlasVectorSearch
        vector_store = MongoDBAtlasVectorSearch(mongodb_client)
        ```

        ```python
        # Create a vector search index programmatically
        vector_store.create_vector_search_index(path="embedding", dimensions=1536, similarity="cosine")

        # Create a text search index programmatically
        vector_store.create_fulltext_search_index("foo)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    _mongodb_client: Any = PrivateAttr()
    _collection: Any = PrivateAttr()
    _vector_index_name: str = PrivateAttr()
    _embedding_key: str = PrivateAttr()
    _id_key: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _metadata_key: str = PrivateAttr()
    _fulltext_index_name: str = PrivateAttr()
    _insert_kwargs: Dict = PrivateAttr()
    _index_name: str = PrivateAttr()  # DEPRECATED
    _oversampling_factor: int = PrivateAttr()

    def __init__(
        self,
        mongodb_client: Optional[Any] = None,
        db_name: str = "default_db",
        collection_name: str = "default_collection",
        vector_index_name: str = "vector_index",
        id_key: str = "_id",
        embedding_key: str = "embedding",
        text_key: str = "text",
        metadata_key: str = "metadata",
        fulltext_index_name: str = "fulltext_index",
        index_name: str = None,
        insert_kwargs: Optional[Dict] = None,
        oversampling_factor: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            mongodb_client: A MongoDB client.
            db_name: A MongoDB database name.
            collection_name: A MongoDB collection name.
            vector_index_name: A MongoDB Atlas *Vector* Search index name. ($vectorSearch)
            id_key: The data field to use as the id.
            embedding_key: A MongoDB field that will contain
            the embedding for each document.
            text_key: A MongoDB field that will contain the text for each document.
            metadata_key: A MongoDB field that will contain
            the metadata for each document.
            insert_kwargs: The kwargs used during `insert`.
            fulltext_index_name: A MongoDB Atlas *full-text* Search index name. ($search)
            oversampling_factor: This times n_results is 'ef' in the HNSW algorithm.
                'ef' determines the number of nearest neighbor candidates to consider during the search phase.
                A higher value leads to more accuracy, but is slower. Default = 10
            index_name: DEPRECATED: Please use vector_index_name.

        """
        super().__init__()

        if mongodb_client is not None:
            self._mongodb_client = cast(MongoClient, mongodb_client)
        else:
            if "MONGODB_URI" not in os.environ:
                raise ValueError(
                    "Must specify MONGODB_URI via env variable "
                    "if not directly passing in client."
                )
            self._mongodb_client = MongoClient(
                os.environ["MONGODB_URI"],
                driver=DriverInfo(name="llama-index", version=version("llama-index")),
            )

        if index_name is not None:
            logger.warning("index_name is deprecated. Please use vector_index_name")
            if vector_index_name is None:
                vector_index_name = index_name
            else:
                logger.warning(
                    "vector_index_name and index_name both specified. Will use vector_index_name"
                )

        self._collection: Collection = self._mongodb_client[db_name][collection_name]
        self._vector_index_name = vector_index_name
        self._embedding_key = embedding_key
        self._id_key = id_key
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._fulltext_index_name = fulltext_index_name
        self._insert_kwargs = insert_kwargs or {}
        self._oversampling_factor = oversampling_factor

        if collection_name not in self._mongodb_client[db_name].list_collection_names():
            self._mongodb_client[db_name].create_collection(collection_name)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

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
        self._collection.delete_many(
            filter={self._metadata_key + ".ref_doc_id": ref_doc_id}, **delete_kwargs
        )

    @property
    def client(self) -> Any:
        """Return MongoDB client."""
        return self._mongodb_client

    @property
    def collection(self) -> Any:
        """Return pymongo Collection."""
        return self._collection

    def _query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        hybrid_top_k = query.hybrid_top_k or query.similarity_top_k
        sparse_top_k = query.sparse_top_k or query.similarity_top_k
        dense_top_k = query.similarity_top_k

        if query.mode == VectorStoreQueryMode.DEFAULT:
            if not query.query_embedding:
                raise ValueError("query_embedding in VectorStoreQueryMode.DEFAULT")
            # Atlas Vector Search, potentially with filter
            logger.debug(f"Running {query.mode} mode query pipeline")
            filter = filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = [
                vector_search_stage(
                    query_vector=query.query_embedding,
                    search_field=self._embedding_key,
                    index_name=self._vector_index_name,
                    limit=dense_top_k,
                    filter=filter,
                    oversampling_factor=self._oversampling_factor,
                ),
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            ]

        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            # Atlas Full-Text Search, potentially with filter
            if not query.query_str:
                raise ValueError("query_str in VectorStoreQueryMode.TEXT_SEARCH ")
            logger.debug(f"Running {query.mode} mode query pipeline")
            filter = filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = fulltext_search_stage(
                query=query.query_str,
                search_field=self._text_key,
                index_name=self._fulltext_index_name,
                operator="text",
                filter=filter,
                limit=sparse_top_k,
            )
            pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})

        elif query.mode == VectorStoreQueryMode.HYBRID:
            # Combines Vector and Full-Text searches with Reciprocal Rank Fusion weighting
            logger.debug(f"Running {query.mode} mode query pipeline")
            scores_fields = ["vector_score", "fulltext_score"]
            filter = filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = []
            # Vector Search pipeline
            if query.query_embedding:
                vector_pipeline = [
                    vector_search_stage(
                        query_vector=query.query_embedding,
                        search_field=self._embedding_key,
                        index_name=self._vector_index_name,
                        limit=dense_top_k,
                        filter=filter,
                        oversampling_factor=self._oversampling_factor,
                    )
                ]
                vector_pipeline.extend(reciprocal_rank_stage("vector_score"))
                combine_pipelines(pipeline, vector_pipeline, self._collection.name)

            # Full-Text Search pipeline
            if query.query_str:
                text_pipeline = fulltext_search_stage(
                    query=query.query_str,
                    search_field=self._text_key,
                    index_name=self._fulltext_index_name,
                    operator="text",
                    filter=filter,
                    limit=sparse_top_k,
                )
                text_pipeline.extend(reciprocal_rank_stage("fulltext_score"))
                combine_pipelines(pipeline, text_pipeline, self._collection.name)

            # Compute weighted sum and sort pipeline
            alpha = (
                query.alpha or 0.5
            )  # If no alpha is given, equal weighting is applied
            pipeline += final_hybrid_stage(
                scores_fields=scores_fields, limit=hybrid_top_k, alpha=alpha
            )

            # Remove embeddings unless requested.
            if (
                query.output_fields is None
                or self._embedding_key not in query.output_fields
            ):
                pipeline.append({"$project": {self._embedding_key: 0}})

        else:
            raise NotImplementedError(
                f"{VectorStoreQueryMode.DEFAULT} (vector), "
                f"{VectorStoreQueryMode.HYBRID} and {VectorStoreQueryMode.TEXT_SEARCH} "
                f"are available. {query.mode} is not."
            )

        # Execution
        logger.debug("Running query pipeline: %s", pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore

        # Post-processing
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            id = res.pop(self._id_key)
            metadata_dict = res.pop(self._metadata_key)

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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        r"""
        Query index for top k most similar nodes.

        The type of search to be performed is based on the VectorStoreQuery.mode.
        Choose from DEFAULT (vector), HYBRID (hybrid), or TEXT_SEARCH (full-text).
        When the mode is one of HYBRID or TEXT_SEARCH,
        VectorStoreQuery.query_str is used for the full-text search.
        See MongoDB Atlas documentation for full details on these.

        For details on VectorStoreQueryMode.DEFAULT == 'default',
        which does vector search, see:
            https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        For details on VectorStoreQueryMode.TEXT_SEARCH == "text_search",
        which performs full-text search, see:
            https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search

        For details on VectorStoreQueryMode.HYBRID == "hybrid",
        which combines the two with Reciprocal Rank Fusion, see the following.
            https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/

        In the scoring algorithm used, Reciprocal Rank Fusion,
            scores := \frac{1}{rank + penalty} with rank in [1,2,..,n]

        Args:
            query: a VectorStoreQuery object.

        Returns:
            A VectorStoreQueryResult containing the results of the query.

        """
        return self._query(query)

    def create_vector_search_index(
        self,
        dimensions: int,
        path: str,
        similarity: str,
        filters: Optional[List[str]] = None,
        *,
        wait_until_complete: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Experimental Utility function to create the vector search index for this store.

        Args:
            dimensions (int): Number of dimensions in embedding
            path (str): field with vector embedding
            similarity (str): The similarity score used for the index
            filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
            wait_until_complete (Optional[float]): If provided, number of seconds to wait
                until search index is ready.
            kwargs: Keyword arguments supplying any additional options to SearchIndexModel.

        """
        return create_vector_search_index(
            self.collection,
            self._vector_index_name,
            dimensions,
            path,
            similarity,
            filters,
            wait_until_complete=wait_until_complete,
            **kwargs,
        )

    def drop_vector_search_index(
        self,
        *,
        wait_until_complete: Optional[float] = None,
    ) -> None:
        """
        Drop the created vector search index for this store.

        Args:
            wait_until_complete (Optional[float]): If provided, number of seconds to wait
                until search index is ready.

        """
        return drop_vector_search_index(
            self.collection,
            self._vector_index_name,
            wait_until_complete=wait_until_complete,
        )

    def update_vector_search_index(
        self,
        dimensions: int,
        path: str,
        similarity: str,
        filters: Optional[List[str]] = None,
        *,
        wait_until_complete: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update the vector search index for this store.

        Replace the existing index definition with the provided definition.

        Args:
            dimensions (int): Number of dimensions in embedding
            path (str): field with vector embedding
            similarity (str): The similarity score used for the index.
            filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
            wait_until_complete (Optional[float]): If provided, number of seconds to wait
                until search index is ready.
            kwargs: Keyword arguments supplying any additional options to SearchIndexModel.

        """
        return update_vector_search_index(
            self.collection,
            self._vector_index_name,
            dimensions,
            path,
            similarity,
            filters,
            wait_until_complete=wait_until_complete,
            **kwargs,
        )

    def create_fulltext_search_index(
        self,
        field: str,
        field_type: str = "string",
        *,
        wait_until_complete: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Experimental Utility function to create the Atlas Search index for this store.

        Args:
            field (str): Field to index
            wait_until_complete (Optional[float]): If provided, number of seconds to wait
                until search index is ready
            kwargs: Keyword arguments supplying any additional options to SearchIndexModel.

        """
        return create_fulltext_search_index(
            self.collection,
            self._fulltext_index_name,
            field,
            field_type,
            wait_until_complete=wait_until_complete,
            **kwargs,
        )
