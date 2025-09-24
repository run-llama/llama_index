import json
import logging
from typing import Any, Dict, Tuple, List, Optional
from .neptune import (
    NeptuneQueryException,
    remove_empty_values,
    create_neptune_analytics_client,
)
from .base_property_graph import (
    NeptuneBasePropertyGraph,
    BASE_ENTITY_LABEL,
    BASE_NODE_LABEL,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.graph_stores.types import LabelledNode, EntityNode, ChunkNode

logger = logging.getLogger(__name__)


class NeptuneAnalyticsPropertyGraphStore(NeptuneBasePropertyGraph):
    supports_vector_queries: bool = True

    def __init__(
        self,
        graph_identifier: str,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new Neptune Analytics graph wrapper instance."""
        self._client = create_neptune_analytics_client(
            graph_identifier, client, credentials_profile_name, region_name
        )
        self.graph_identifier = graph_identifier

    def structured_query(self, query: str, param_map: Dict[str, Any] = None) -> Any:
        """
        Run the structured query.

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
            resp = self.client.execute_query(
                graphIdentifier=self.graph_identifier,
                queryString=query,
                parameters=param_map,
                language="OPEN_CYPHER",
            )
            return json.loads(resp["payload"].read().decode("UTF-8"))["results"]
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
        """
        Query the graph store with a vector store query.

        Returns:
            (nodes, score): The nodes and their associated score

        """
        conditions = None
        if query.filters:
            conditions = [
                f"e.{filter.key} {filter.operator.value} {filter.value}"
                for filter in query.filters.filters
            ]
        filters = (
            f" {query.filters.condition.value} ".join(conditions).replace("==", "=")
            if conditions is not None
            else "1 = 1"
        )

        data = self.structured_query(
            f"""MATCH (e:`{BASE_ENTITY_LABEL}`)
            WHERE ({filters})
            CALL neptune.algo.vectors.get(e)
            YIELD embedding
            WHERE embedding IS NOT NULL
            CALL neptune.algo.vectors.topKByNode(e)
            YIELD node, score
            WITH e, score
            ORDER BY score DESC LIMIT $limit
            RETURN e.id AS name,
                [l in labels(e) WHERE l <> '{BASE_ENTITY_LABEL}' | l][0] AS type,
                e{{.* , embedding: Null, name: Null, id: Null}} AS properties,
                score""",
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )
        data = data if data else []

        nodes = []
        scores = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=remove_empty_values(record["properties"]),
            )
            nodes.append(node)
            scores.append(record["score"])

        return (nodes, scores)

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """
        Upsert the nodes in the graph.

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
                    WITH c, row.embedding as e
                    WHERE e IS NOT NULL
                    CALL neptune.algo.vectors.upsert(c, e)
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
                    SET e:`{d["label"]}`
                    WITH e, row
                    WHERE removeKeyFromMap(row.properties, '').triplet_source_id IS NOT NULL
                    MERGE (c:Chunk {{id: removeKeyFromMap(row.properties, '').triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    WITH e, row.embedding as em
                    CALL neptune.algo.vectors.upsert(e, em)
                    RETURN count(*) as count
                    """,
                    param_map={"data": d},
                )

    def _get_summary(self) -> Dict:
        """
        Get the Summary of the graph topology.

        Returns:
            Dict: The graph summary

        """
        try:
            response = self.client.get_graph_summary(
                graphIdentifier=self.graph_identifier, mode="detailed"
            )
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": ("Summary API error occurred on Neptune Analytics"),
                    "details": str(e),
                }
            )

        try:
            summary = response["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
