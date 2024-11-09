"""Memgraph graph store index."""
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.graph_stores.types import GraphStore

logger = logging.getLogger(__name__)

node_properties_query = """
CALL schema.node_type_properties()
YIELD nodeType AS label, propertyName AS property, propertyTypes AS type
WITH label AS nodeLabels, collect({property: property, type: type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output
"""

rel_properties_query = """
CALL schema.rel_type_properties()
YIELD relType AS label, propertyName AS property, propertyTypes AS type
WITH label, collect({property: property, type: type}) AS properties
RETURN {type: label, properties: properties} AS output
"""

rel_query = """
MATCH (start_node)-[r]->(end_node)
WITH labels(start_node) AS start, type(r) AS relationship_type, labels(end_node) AS end, keys(r) AS relationship_properties
UNWIND end AS end_label
RETURN DISTINCT {start: start[0], type: relationship_type, end: end_label} AS output
"""


class MemgraphGraphStore(GraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "memgraph",
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError("Please install neo4j: pip install neo4j")
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
        # verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Memgraph database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Memgraph database. "
                "Please ensure that the username and password are correct"
            )
        # set schema
        self.refresh_schema()

        # create constraint
        self.query(
            """
            CREATE CONSTRAINT ON (n:%s) ASSERT n.id IS UNIQUE;
            """
            % (self.node_label)
        )

        # create index
        self.query(
            """
            CREATE INDEX ON :%s(id);
            """
            % (self.node_label)
        )

    @property
    def client(self) -> Any:
        return self._driver

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Execute a Cypher query."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [record.data() for record in result]

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = f"""
            MATCH (n1:{self.node_label})-[r]->(n2:{self.node_label})
            WHERE n1.id = $subj
            RETURN type(r), n2.id;
        """

        with self._driver.session(database=self._database) as session:
            data = session.run(query, {"subj": subj})
            return [record.values() for record in data]

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get flat relation map."""
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            return rel_map

        query = (
            f"""MATCH p=(n1:{self.node_label})-[*1..{depth}]->() """
            f"""{"WHERE n1.id IN $subjs" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1.id AS subj, collect([type(rel), endNode(rel).id]) AS rels "
            "RETURN subj, rels"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            rel_map[record["subj"]] = record["rels"]

        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = f"""
            MERGE (n1:`{self.node_label}` {{id:$subj}})
            MERGE (n2:`{self.node_label}` {{id:$obj}})
            MERGE (n1)-[:`{rel.replace(" ", "_").upper()}`]->(n2)
        """
        self.query(query, {"subj": subj, "obj": obj})

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        query = f"""
            MATCH (n1:`{self.node_label}`)-[r:`{rel}`]->(n2:`{self.node_label}`)
            WHERE n1.id = $subj AND n2.id = $obj
            DELETE r
        """
        self.query(query, {"subj": subj, "obj": obj})

    def refresh_schema(self) -> None:
        """
        Refreshes the Memgraph graph schema information.
        """
        node_properties = self.query(node_properties_query)
        relationships_properties = self.query(rel_properties_query)
        relationships = self.query(rel_query)

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {relationships_properties}
        The relationships are the following:
        {relationships}
        """

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the MemgraphGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema
