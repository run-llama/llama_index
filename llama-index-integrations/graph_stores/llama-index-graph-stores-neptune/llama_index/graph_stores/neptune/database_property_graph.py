import json
import logging

from typing import Any, Dict, Tuple, List, Optional
from .neptune import NeptuneQueryException, create_neptune_database_client
from .base_property_graph import (
    NeptuneBasePropertyGraph,
    BASE_ENTITY_LABEL,
    BASE_NODE_LABEL,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.graph_stores.types import LabelledNode, EntityNode, ChunkNode

logger = logging.getLogger(__name__)


class NeptuneDatabasePropertyGraphStore(NeptuneBasePropertyGraph):
    supports_vector_queries: bool = False

    def __init__(
        self,
        host: str,
        port: int = 8182,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        sign: bool = True,
        use_https: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init.

        Args:
            host (str): The host endpoint
            port (int, optional): The port. Defaults to 8182.
            client (Any, optional): If provided, this is the client that will be used. Defaults to None.
            credentials_profile_name (Optional[str], optional): If provided this is the credentials profile that will be used. Defaults to None.
            region_name (Optional[str], optional): The region to use. Defaults to None.
            sign (bool, optional): True will SigV4 sign all requests, False will not. Defaults to True.
            use_https (bool, optional): True to use https, False to use http. Defaults to True.
        """
        self._client = create_neptune_database_client(
            host, port, client, credentials_profile_name, region_name, sign, use_https
        )

    def structured_query(self, query: str, param_map: Dict[str, Any] = None) -> Any:
        """Run the structured query.

        Args:
            query (str): The query to run
            param_map (Dict[str, Any] | None, optional): A dictionary of query parameters. Defaults to None.

        Raises:
            NeptuneQueryException: An exception from Neptune with details

        Returns:
            Any: The results of the query
        """
        param_map = param_map or {}

        try:
            logger.debug(
                f"structured_query() query: {query} parameters: {json.dumps(param_map)}"
            )

            return self.client.execute_open_cypher_query(
                openCypherQuery=query, parameters=json.dumps(param_map)
            )["results"]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                    "query": query,
                    "parameters": str(param_map),
                }
            )

    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> Tuple[List[Any]]:
        """NOT SUPPORTED.

        Args:
            query (VectorStoreQuery): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[List[LabelledNode] | List[float]]: _description_
        """
        raise NotImplementedError

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """Upsert the nodes in the graph.

        Args:
            nodes (List[LabelledNode]): The list of nodes to upsert
        """
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            for d in chunk_dicts:
                self.structured_query(
                    """
                    WITH $data AS row
                    MERGE (c:Chunk {id: row.id})
                    SET c.text = row.text
                    SET c += removeKeyFromMap(row.properties, '')
                    RETURN count(*)
                    """,
                    param_map={"data": d},
                )

        if entity_dicts:
            for d in entity_dicts:
                self.structured_query(
                    f"""
                    WITH $data AS row
                    MERGE (e:`{BASE_NODE_LABEL}` {{id: row.id}})
                    SET e += removeKeyFromMap(row.properties, '')
                    SET e.name = row.name, e:`{BASE_ENTITY_LABEL}`
                    SET e:`{d['label']}`
                    WITH e, row
                    WHERE removeKeyFromMap(row.properties, '').triplet_source_id IS NOT NULL
                    MERGE (c:Chunk {{id: removeKeyFromMap(row.properties, '').triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    RETURN count(*) as count
                    """,
                    param_map={"data": d},
                )

    def _get_summary(self) -> Dict:
        """Get the Summary of the graph schema.

        Returns:
            Dict: The graph summary
        """
        try:
            response = self.client.get_propertygraph_summary()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": (
                        "Summary API is not available for this instance of Neptune,"
                        "ensure the engine version is >=1.2.1.0"
                    ),
                    "details": str(e),
                }
            )

        try:
            summary = response["payload"]["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
