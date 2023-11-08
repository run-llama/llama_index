from __future__ import annotations

from enum import Enum
from os import getenv
from time import sleep
from types import ModuleType
from typing import Any, List, Type, TypeVar

from llama_index.schema import BaseNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    DEFAULT_EMBEDDING_KEY,
    DEFAULT_TEXT_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

T = TypeVar("T", bound="RocksetVectorStore")


def _get_rockset() -> ModuleType:
    """Gets the rockset module and raises an ImportError if
    the rockset package hasn't been installed.

    Returns:
        rockset module (ModuleType)
    """
    try:
        import rockset
    except ImportError:
        raise ImportError("Please install rockset with `pip install rockset`")
    return rockset


def _get_client(api_key: str | None, api_server: str | None, client: Any | None) -> Any:
    """Returns the passed in client object if valid, else
    constructs and returns one.

    Returns:
        The rockset client object (rockset.RocksetClient)
    """
    rockset = _get_rockset()
    if client:
        if type(client) is not rockset.RocksetClient:
            raise ValueError("Parameter `client` must be of type rockset.RocksetClient")
    elif not api_key and not getenv("ROCKSET_API_KEY"):
        raise ValueError(
            "Parameter `client`, `api_key` or env var `ROCKSET_API_KEY` must be set"
        )
    else:
        client = rockset.RocksetClient(
            api_key=api_key or getenv("ROCKSET_API_KEY"),
            host=api_server or getenv("ROCKSET_API_SERVER"),
        )
    return client


class RocksetVectorStore(VectorStore):
    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    class DistanceFunc(Enum):
        COSINE_SIM = "COSINE_SIM"
        EUCLIDEAN_DIST = "EUCLIDEAN_DIST"
        DOT_PRODUCT = "DOT_PRODUCT"

    def __init__(
        self,
        collection: str,
        client: Any | None = None,
        text_key: str = DEFAULT_TEXT_KEY,
        embedding_col: str = DEFAULT_EMBEDDING_KEY,
        metadata_col: str = "metadata",
        workspace: str = "commons",
        api_server: str | None = None,
        api_key: str | None = None,
        distance_func: DistanceFunc = DistanceFunc.COSINE_SIM,
    ) -> None:
        """Rockset Vector Store Data container.

        Args:
            collection (str): The name of the collection of vectors
            client (Optional[Any]): Rockset client object
            text_key (str): The key to the text of nodes
                (default: llama_index.vector_stores.utils.DEFAULT_TEXT_KEY)
            embedding_col (str): The DB column containing embeddings
                (default: llama_index.vector_stores.utils.DEFAULT_EMBEDDING_KEY))
            metadata_col (str): The DB column containing node metadata
                (default: "metadata")
            workspace (str): The workspace containing the collection of vectors
                (default: "commons")
            api_server (Optional[str]): The Rockset API server to use
            api_key (Optional[str]): The Rockset API key to use
            distance_func (RocksetVectorStore.DistanceFunc): The metric to measure
                vector relationship
                (default: RocksetVectorStore.DistanceFunc.COSINE_SIM)
        """
        self.rockset = _get_rockset()
        self.rs = _get_client(api_key, api_server, client)
        self.workspace = workspace
        self.collection = collection
        self.text_key = text_key
        self.embedding_col = embedding_col
        self.metadata_col = metadata_col
        self.distance_func = distance_func
        self.distance_order = (
            "ASC" if distance_func is distance_func.EUCLIDEAN_DIST else "DESC"
        )

        try:
            self.rs.set_application("llama_index")
        except AttributeError:
            # set_application method does not exist.
            # rockset version < 2.1.0
            pass

    @property
    def client(self) -> Any:
        return self.rs

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Stores vectors in the collection.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            Stored node IDs (List[str])
        """
        return [
            row["_id"]
            for row in self.rs.Documents.add_documents(
                collection=self.collection,
                workspace=self.workspace,
                data=[
                    {
                        self.embedding_col: node.get_embedding(),
                        "_id": node.node_id,
                        self.metadata_col: node_to_metadata_dict(
                            node, text_field=self.text_key
                        ),
                    }
                    for node in nodes
                ],
            ).data
        ]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Deletes nodes stored in the collection by their ref_doc_id.

        Args:
            ref_doc_id (str): The ref_doc_id of the document
                whose nodes are to be deleted
        """
        self.rs.Documents.delete_documents(
            collection=self.collection,
            workspace=self.workspace,
            data=[
                self.rockset.models.DeleteDocumentsRequestData(id=row["_id"])
                for row in self.rs.sql(
                    f"""
                        SELECT
                            _id
                        FROM
                            "{self.workspace}"."{self.collection}" x
                        WHERE
                            x.{self.metadata_col}.ref_doc_id=:ref_doc_id
                    """,
                    params={"ref_doc_id": ref_doc_id},
                ).results
            ],
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Gets nodes relevant to a query.

        Args:
            query (llama_index.vector_stores.types.VectorStoreQuery): The query
            similarity_col (Optional[str]): The column to select the cosine
                similarity as (default: "_similarity")

        Returns:
            query results (llama_index.vector_stores.types.VectorStoreQueryResult)
        """
        similarity_col = kwargs.get("similarity_col", "_similarity")
        res = self.rs.sql(
            f"""
                SELECT
                    _id,
                    {self.metadata_col}
                    {
                        f''', {self.distance_func.value}(
                            {query.query_embedding},
                            {self.embedding_col}
                        )
                            AS {similarity_col}'''
                        if query.query_embedding
                        else ''
                    }
                FROM
                    "{self.workspace}"."{self.collection}" x
                {"WHERE" if query.node_ids or query.filters else ""} {
                    f'''({
                        ' OR '.join([
                            f"_id='{node_id}'" for node_id in query.node_ids
                        ])
                    })''' if query.node_ids else ""
                } {
                    f''' {'AND' if query.node_ids else ''} ({
                        ' AND '.join([
                            f"x.{self.metadata_col}.{filter.key}=:{filter.key}"
                            for filter
                            in query.filters.filters
                        ])
                    })''' if query.filters else ""
                }
                ORDER BY
                    {similarity_col} {self.distance_order}
                LIMIT
                    {query.similarity_top_k}
            """,
            params={filter.key: filter.value for filter in query.filters.filters}
            if query.filters
            else {},
        )

        similarities: List[float] | None = [] if query.query_embedding else None
        nodes, ids = [], []
        for row in res.results:
            if similarities is not None:
                similarities.append(row[similarity_col])
            nodes.append(metadata_dict_to_node(row[self.metadata_col]))
            ids.append(row["_id"])

        return VectorStoreQueryResult(similarities=similarities, nodes=nodes, ids=ids)

    @classmethod
    def with_new_collection(
        cls: Type[T], dimensions: int | None = None, **rockset_vector_store_args: Any
    ) -> RocksetVectorStore:
        """Creates a new collection and returns its RocksetVectorStore.

        Args:
            dimensions (Optional[int]): The length of the vectors to enforce
                in the collection's ingest transformation. By default, the
                collection will do no vector enforcement.
            collection (str): The name of the collection to be created
            client (Optional[Any]): Rockset client object
            workspace (str): The workspace containing the collection to be
                created (default: "commons")
            text_key (str): The key to the text of nodes
                (default: llama_index.vector_stores.utils.DEFAULT_TEXT_KEY)
            embedding_col (str): The DB column containing embeddings
                (default: llama_index.vector_stores.utils.DEFAULT_EMBEDDING_KEY))
            metadata_col (str): The DB column containing node metadata
                (default: "metadata")
            api_server (Optional[str]): The Rockset API server to use
            api_key (Optional[str]): The Rockset API key to use
            distance_func (RocksetVectorStore.DistanceFunc): The metric to measure
                vector relationship
                (default: RocksetVectorStore.DistanceFunc.COSINE_SIM)
        """
        client = rockset_vector_store_args["client"] = _get_client(
            api_key=rockset_vector_store_args.get("api_key"),
            api_server=rockset_vector_store_args.get("api_server"),
            client=rockset_vector_store_args.get("client"),
        )
        collection_args = {
            "workspace": rockset_vector_store_args.get("workspace", "commons"),
            "name": rockset_vector_store_args.get("collection"),
        }
        embeddings_col = rockset_vector_store_args.get(
            "embeddings_col", DEFAULT_EMBEDDING_KEY
        )
        if dimensions:
            collection_args[
                "field_mapping_query"
            ] = _get_rockset().model.field_mapping_query.FieldMappingQuery(
                sql=f"""
                    SELECT
                        *, VECTOR_ENFORCE(
                            {embeddings_col},
                            {dimensions},
                            'float'
                        ) AS {embeddings_col}
                    FROM
                        _input
                """
            )

        client.Collections.create_s3_collection(**collection_args)  # create collection
        while (
            client.Collections.get(
                collection=rockset_vector_store_args.get("collection")
            ).data.status
            != "READY"
        ):  # wait until collection is ready
            sleep(0.1)
            # TODO: add async, non-blocking method collection creation

        return cls(
            **dict(
                filter(  # filter out None args
                    lambda arg: arg[1] is not None, rockset_vector_store_args.items()
                )
            )
        )
