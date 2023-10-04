"""Simple graph store index."""

import logging
from typing import Any, Dict, List, Optional

from llama_index.graph_stores.types import GraphStore

logger = logging.getLogger(__name__)


class FalkorDBGraphStore(GraphStore):
    """FalkorDB Graph Store.

    In this graph store, triplets are stored within FalkorDB.

    Args:
        simple_graph_store_data_dict (Optional[dict]): data dict
            containing the triplets. See FalkorDBGraphStoreData
            for more details.
    """

    def __init__(
        self,
        url: str,
        database: str = "falkor",
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError("Please install redis client: pip install redis")

        """Initialize params."""
        self._node_label = node_label

        self._driver = redis.Redis.from_url(url).graph(database)
        self._driver.query(f"CREATE INDEX FOR (n:`{self._node_label}`) ON (n.id)")

        self._database = database

        self.schema = ""
        self.get_query = f"""
            MATCH (n1:`{self._node_label}`)-[r]->(n2:`{self._node_label}`)
            WHERE n1.id = $subj RETURN type(r), n2.id
        """

    @property
    def client(self) -> None:
        return self._driver

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        result = self._driver.query(
            self.get_query, params={"subj": subj}, read_only=True
        )
        return result.result_set

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        query = f"""
            MATCH (n1:{self._node_label})
            WHERE n1.id IN $subjs
            WITH n1
            MATCH p=(n1)-[e*1..{depth}]->(z)
            RETURN p LIMIT {limit}
        """

        data = self.query(query, params={"subjs": subjs})
        if not data:
            return rel_map

        for record in data:
            nodes = record[0].nodes()
            edges = record[0].edges()

            subj_id = nodes[0].properties["id"]
            path = []
            for i, edge in enumerate(edges):
                dest = nodes[i + 1]
                dest_id = dest.properties["id"]
                path.append(edge.relation)
                path.append(dest_id)

            paths = rel_map[subj_id] if subj_id in rel_map else []
            paths.append(path)
            rel_map[subj_id] = paths

        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[:`%s`]->(n2)
        """

        prepared_statement = query % (
            self._node_label,
            self._node_label,
            rel.replace(" ", "_").upper(),
        )

        # Call FalkorDB with prepared statement
        self._driver.query(prepared_statement, params={"subj": subj, "obj": obj})

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            rel = rel.replace(" ", "_").upper()
            query = f"""
                MATCH (n1:`{self._node_label}`)-[r:`{rel}`]->(n2:`{self._node_label}`)
                WHERE n1.id = $subj AND n2.id = $obj DELETE r
            """

            # Call FalkorDB with prepared statement
            self._driver.query(query, params={"subj": subj, "obj": obj})

        def delete_entity(entity: str) -> None:
            query = f"MATCH (n:`{self._node_label}`) WHERE n.id = $entity DELETE n"

            # Call FalkorDB with prepared statement
            self._driver.query(query, params={"entity": entity})

        def check_edges(entity: str) -> bool:
            query = f"""
                MATCH (n1:`{self._node_label}`)--()
                WHERE n1.id = $entity RETURN count(*)
            """

            # Call FalkorDB with prepared statement
            result = self._driver.query(
                query, params={"entity": entity}, read_only=True
            )
            return bool(result.result_set)

        delete_rel(subj, obj, rel)
        if not check_edges(subj):
            delete_entity(subj)
        if not check_edges(obj):
            delete_entity(obj)

    def refresh_schema(self) -> None:
        """
        Refreshes the FalkorDB graph schema information.
        """
        node_properties = self.query("CALL DB.PROPERTYKEYS()")
        relationships = self.query("CALL DB.RELATIONSHIPTYPES()")

        self.schema = f"""
        Properties: {node_properties}
        Relationships: {relationships}
        """

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the FalkorDBGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        result = self._driver.query(query, params=params)
        return result.result_set
