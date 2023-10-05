"""Neo4j graph store index."""
import logging
from typing import Any, Dict, List, Optional

from llama_index.graph_stores.types import GraphStore

logger = logging.getLogger(__name__)

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
RETURN "(:" + label + ")-[:" + property + "]->(:" + toString(other_node) + ")" AS output
"""


class Neo4jGraphStore(GraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
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
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        # Set schema
        try:
            self.refresh_schema()
        except neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )
        # Create constraint for faster insert and retrieval
        try:  # Using Neo4j 5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:%s) REQUIRE n.id IS UNIQUE;
                """
                % (self.node_label)
            )
        except Exception:  # Using Neo4j <5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS ON (n:%s) ASSERT n.id IS UNIQUE;
                """
                % (self.node_label)
            )

    @property
    def client(self) -> Any:
        return self._driver

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r]->(n2:%s)
            WHERE n1.id = $subj
            RETURN type(r), n2.id;
        """

        prepared_statement = query % (self.node_label, self.node_label)

        with self._driver.session(database=self._database) as session:
            data = session.run(prepared_statement, {"subj": subj})
            return [record.values() for record in data]

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

        query = (
            f"""MATCH p=(n1:{self.node_label})-[*1..{depth}]->() """
            f"""{"WHERE n1.id IN $subjs" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1.id AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            "collect([type(rel), endNode(rel).id]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            rel_map[record["subj"]] = record["flattened_rels"]
        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[:`%s`]->(n2)
        """

        prepared_statement = query % (
            self.node_label,
            self.node_label,
            rel.replace(" ", "_").upper(),
        )

        with self._driver.session(database=self._database) as session:
            session.run(prepared_statement, {"subj": subj, "obj": obj})

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            with self._driver.session(database=self._database) as session:
                session.run(
                    (
                        "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.id = $subj AND n2.id"
                        " = $obj DELETE r"
                    ).format(self.node_label, rel, self.node_label),
                    {"subj": subj, "obj": obj},
                )

        def delete_entity(entity: str) -> None:
            with self._driver.session(database=self._database) as session:
                session.run(
                    "MATCH (n:%s) WHERE n.id = $entity DELETE n" % self.node_label,
                    {"entity": entity},
                )

        def check_edges(entity: str) -> bool:
            with self._driver.session(database=self._database) as session:
                is_exists_result = session.run(
                    "MATCH (n1:%s)--() WHERE n1.id = $entity RETURN count(*)"
                    % (self.node_label),
                    {"entity": entity},
                )
                return bool(list(is_exists_result))

        delete_rel(subj, obj, rel)
        if not check_edges(subj):
            delete_entity(subj)
        if not check_edges(obj):
            delete_entity(obj)

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        node_properties = self.query(node_properties_query)
        relationships_properties = self.query(rel_properties_query)
        relationships = self.query(rel_query)

        self.schema = f"""
        Node properties are the following:
        {[el['output'] for el in node_properties]}
        Relationship properties are the following:
        {[el['output'] for el in relationships_properties]}
        The relationships are the following:
        {[el['output'] for el in relationships]}
        """

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the Neo4jGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]
