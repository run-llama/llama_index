from os import getenv
from time import sleep
from enum import Enum
from typing import List, Any, Optional
from types import ModuleType
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    DEFAULT_TEXT_KEY,
    DEFAULT_EMBEDDING_KEY,
)

def _get_rockset() -> ModuleType:
    """Gets the rockset module and raises an ImportError if
    the rockset package hasn't been installed

    Returns
        rockset module (ModuleType)
    """
    try:
        import rockset
    except ImportError:
        raise ImportError("Please install rockset with `pip install rockset`")
    return rockset

def _assert_client_type(client: Any, rockset: ModuleType) -> None:
    """Raises a ValueError if client is not of type rockset.RocksetClient
    Args:
        client (Any): The RocksetClient object
        rockset (Any): The rockset module
    """
    if not type(client) is rockset.RocksetClient:
        raise ValueError("Parameter `client` must be of type rockset.RocksetClient")

class RocksetVectorStore(VectorStore):
    stores_text: bool = True
    is_embedding_query: bool = True

    class DistanceFunc(Enum):
        COSINE_SIM = "COSINE_SIM"
        EUCLIDEAN_DIST = "EUCLIDEAN_DIST"
        DOT_PRODUCT = "DOT_PRODUCT"

    def __init__(
        self,
        collection: str,
        client: Optional[Any] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        embedding_col: str = DEFAULT_EMBEDDING_KEY,
        metadata_col: str = "metadata",
        workspace: str = "commons",
        api_server: Optional[str] = None,
        api_key: Optional[str] = None,
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
            workspace (str): The workspace containing the colection of vectors
                (default: "commons")
            api_server (Optional[str]): The Rockset API server to use
            api_key (Optional[str]): The Rockset API key to use
            distance_func (RocksetVectorStore.DistanceFunc): The metric to measure
                vector relationship
                (default: RocksetVectorStore.DistanceFunc.COSINE_SIM)
        """
        self.rockset = _get_rockset()
        if client and not type(client) is self.rockset.RocksetClient:
            raise ValueError("Parameter `client` must be of type rockset.RocksetClient")
        try:
            self.rs = client or self.rockset.RocksetClient(
                host=api_server
                or getenv("ROCKSET_API_SERVER")
                or "https://api.usw2a1.rockset.com",
                api_key=api_key or getenv("ROCKSET_API_KEY"),
            )
        except self.rockset.exceptions.InitializationException:
            raise ValueError(
                "Must either pass in `client`, `api_key`, or set ROCKSET_API_KEY env var"
            )
        self.workspace = workspace
        self.collection = collection
        self.text_key = text_key
        self.embedding_col = embedding_col
        self.metadata_col = metadata_col
        self.distance_func = distance_func
        self.distance_order = (
            "ASC" if distance_func is distance_func.EUCLIDEAN_DIST else "DESC"
        )

    @property
    def client(self) -> Any:
        return self.rs

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Stores vectors in the collection

        Args:
            embedding_results (List[NodeWithEmbedding]): The embedding nodes to store

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
                        self.embedding_col: result.embedding,
                        "_id": result.id,
                        self.metadata_col: node_to_metadata_dict(
                            result.node, text_field=self.text_key
                        ),
                    }
                    for result in embedding_results
                ],
            ).data
        ]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Deletes nodes stored in the collection by their ref_doc_id

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
                            x.{self.metadata_col}.ref_doc_id=:doc_id
                    """,
                    params={"doc_id": ref_doc_id},
                ).results
            ],
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Gets nodes relevant to a query

        Args:
            query (llama_index.vector_stores.types.VectorStoreQuery): The query
            similarity_col (Optional[str]): The column to select the cosine
                similarity as (default: "_similarity")

        Returns:
            query results (llama_index.vector_stores.types.VectorStoreQueryResult)
        """
        similarity_col = kwargs.get("similarity_col") or "_similarity"
        res = self.rs.sql(
            f"""
                SELECT 
                    _id, 
                    {self.distance_func.value}({query.query_embedding}, {self.embedding_col}) AS {similarity_col},
                    {self.metadata_col}
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
                            f'x.{self.metadata_col}.{filter.key}=:{filter.key}' for filter in query.filters.filters
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

        similarities, nodes, ids = [], [], []
        for row in res.results:
            similarities.append(row[similarity_col])
            nodes.append(metadata_dict_to_node(row[self.metadata_col]))
            ids.append(row["_id"])

        return VectorStoreQueryResult(similarities=similarities, nodes=nodes, ids=ids)
    
    @classmethod
    def with_new_collection(
        cls, 
        client: Any,
        collection_name: str, 
        workspace: str="commons",
        text_key: str = DEFAULT_TEXT_KEY,
        embedding_col: str = DEFAULT_EMBEDDING_KEY,
        metadata_col: str = "metadata",
        distance_func: DistanceFunc = DistanceFunc.COSINE_SIM,
    ):
        _assert_client_type(client, _get_rockset()) # raise err if client is not the right type
        client.Collections.create_s3_collection(
            workspace=workspace, 
            name=collection_name
        ) # create collection
        while not client.Collections.get(collection=collection_name).data.status == "READY": # wait until collection is ready
            sleep(0.1)
        return cls(
            collection_name, 
            client=client,
            text_key=text_key,
            embedding_col=embedding_col,
            metadata_col=metadata_col,
            distance_func=distance_func
        )