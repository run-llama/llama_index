from typing import Any, Dict, List, Optional

import json
import logging
from typing import Union

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)


logger = logging.getLogger(__name__)


class NeptuneVectorQueryException(Exception):
    """Exception for the Neptune queries."""

    def __init__(self, exception: Union[str, Dict]):
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class NeptuneAnalyticsVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    node_label: str
    graph_identifier: str
    embedding_dimension: int
    text_node_property: str
    hybrid_search: bool
    retrieval_query: Optional[str]

    _client: Any = PrivateAttr()

    def __init__(
        self,
        graph_identifier: str,
        embedding_dimension: int,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        hybrid_search: bool = False,
        node_label: str = "Chunk",
        text_node_property: str = "text",
        retrieval_query: str = None,
        **kwargs: Any,
    ) -> None:
        """Create a new Neptune Analytics graph wrapper instance."""
        super().__init__(
            graph_identifier=graph_identifier,
            embedding_dimension=embedding_dimension,
            node_label=node_label,
            text_node_property=text_node_property,
            hybrid_search=hybrid_search,
            retrieval_query=retrieval_query,
        )

        try:
            if client is not None:
                self._client = client
            else:
                import boto3

                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                if region_name:
                    self._client = session.client(
                        "neptune-graph", region_name=region_name
                    )
                else:
                    self._client = session.client("neptune-graph")

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "NeptuneGraph requires a boto3 version 1.34.40 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        # Verify that the analytics graph has a vector search index and that the dimensions match
        self._verify_vectorIndex()

    def _verify_vectorIndex(self) -> None:
        """
        Check if the connected Neptune Analytics graph has VSS enabled and that the dimensions are the same.
        """
        resp = self._client.get_graph(graphIdentifier=self.graph_identifier)

        if "vectorSearchConfiguration" in resp:
            if (
                not resp["vectorSearchConfiguration"]["dimension"]
                == self.embedding_dimension
            ):
                raise ValueError(
                    f"Vector search index dimension for Neptune Analytics graph does not match the provided value."
                )
        else:
            raise ValueError(
                f"Vector search index does not exist for the Neptune Analytics graph."
            )

    @classmethod
    def class_name(cls) -> str:
        return "NeptuneAnalyticsVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def database_query(
        self, query: str, params: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        This method sends a query to the Neptune Analytics graph
        and returns the results as a list of dictionaries.

        Args:
            query (str): The openCypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        try:
            logger.debug(f"query() query: {query} parameters: {json.dumps(params)}")
            resp = self._client.execute_query(
                graphIdentifier=self.graph_identifier,
                queryString=query,
                parameters=params,
                language="OPEN_CYPHER",
            )
            return json.loads(resp["payload"].read().decode("UTF-8"))["results"]
        except Exception as e:
            raise NeptuneVectorQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                }
            )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids = [r.node_id for r in nodes]

        for r in nodes:
            import_query = (
                f"MERGE (c:`{self.node_label}` {{`~id`: $id}}) "
                "SET c += $data "
                "WITH c "
                f"CALL neptune.algo.vectors.upsert(c, {r.embedding}) "
                "YIELD node "
                "RETURN id(node) as id"
            )
            resp = self.database_query(
                import_query,
                params=self.__clean_params(r),
            )
        print("Nodes added")
        return ids

    def _get_search_index_query(self, hybrid: bool, k: int = 10) -> str:
        if not hybrid:
            return (
                "WITH $embedding as emb "
                "CALL neptune.algo.vectors.topKByEmbedding(emb, {topK: "
                + str(k)
                + "}) YIELD embedding, node, score "
            )
        else:
            raise NotImplementedError

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        default_retrieval = (
            f"RETURN node.`{self.text_node_property}` AS text, score, "
            "id(node) AS id, "
            f"node AS metadata"
        )

        retrieval_query = self.retrieval_query or default_retrieval
        read_query = (
            self._get_search_index_query(self.hybrid_search, query.similarity_top_k)
            + retrieval_query
        )

        parameters = {
            "embedding": query.query_embedding,
        }

        results = self.database_query(read_query, params=parameters)

        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = metadata_dict_to_node(record["metadata"]["~properties"])
            node.set_content(str(record["text"]))
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.database_query(
            f"MATCH (n:`{self.node_label}`) WHERE n.ref_doc_id = $id DETACH DELETE n",
            params={"id": ref_doc_id},
        )

    def __clean_params(self, record: BaseNode) -> List[Dict[str, Any]]:
        """Convert BaseNode object to a dictionary to be imported into Neo4j."""
        text = record.get_content(metadata_mode=MetadataMode.NONE)
        id = record.node_id
        metadata = node_to_metadata_dict(record, remove_text=True, flat_metadata=False)
        # Remove redundant metadata information
        for k in ["document_id", "doc_id"]:
            del metadata[k]
        return {"id": id, "data": {self.text_node_property: text, "id": id, **metadata}}
