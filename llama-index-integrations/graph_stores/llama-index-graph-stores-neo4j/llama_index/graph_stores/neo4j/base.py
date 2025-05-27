"""Neo4j graph store index."""

import logging
from typing import Any, Dict, List, Optional, Type
from types import TracebackType

from llama_index.core.graph_stores.types import GraphStore

import neo4j

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
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""


class Neo4jGraphStore(GraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        node_label: str = "Entity",
        refresh_schema: bool = True,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self._timeout = timeout
        self.schema = ""
        self.structured_schema: Dict[str, Any] = {}
        # Verify connection
        try:
            with self._driver as driver:
                driver.verify_connectivity()
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
        self.schema = ""
        self.structured_schema = {}
        if refresh_schema:
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
            f"""WHERE toLower(n1.id) IN {[subj.lower() for subj in subjs] if subjs else []}"""
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
        node_properties = [el["output"] for el in self.query(node_properties_query)]
        rel_properties = [el["output"] for el in self.query(rel_properties_query)]
        relationships = [el["output"] for el in self.query(rel_query)]

        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in rel_properties},
            "relationships": relationships,
        }

        # Format node properties
        formatted_node_props = []
        for el in node_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_node_props.append(f"{el['labels']} {{{props_str}}}")

        # Format relationship properties
        formatted_rel_props = []
        for el in rel_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_rel_props.append(f"{el['type']} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships
        ]

        self.schema = "\n".join(
            [
                "Node properties are the following:",
                ",".join(formatted_node_props),
                "Relationship properties are the following:",
                ",".join(formatted_rel_props),
                "The relationships are the following:",
                ",".join(formatted_rels),
            ]
        )

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the Neo4jGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        param_map = param_map or {}
        try:
            data, _, _ = self._driver.execute_query(
                neo4j.Query(text=query, timeout=self._timeout),
                database_=self._database,
                parameters_=param_map,
            )
            return [r.data() for r in data]
        except neo4j.exceptions.Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                )
            ):
                raise
        # Fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            data = session.run(
                neo4j.Query(text=query, timeout=self._timeout), param_map
            )
            return [r.data() for r in data]

    def close(self) -> None:
        """
        Explicitly close the Neo4j driver connection.

        Delegates connection management to the Neo4j driver.
        """
        if hasattr(self, "_driver"):
            self._driver.close()
            # Remove the driver attribute to indicate closure
            delattr(self, "_driver")

    def __enter__(self) -> "Neo4jGraphStore":
        """
        Enter the runtime context for the Neo4j graph connection.

        Enables use of the graph connection with the 'with' statement.
        This method allows for automatic resource management and ensures
        that the connection is properly handled.

        Returns:
            Neo4jPropertyGraphStore: The current graph connection instance

        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the runtime context for the Neo4j graph connection.

        This method is automatically called when exiting a 'with' statement.
        It ensures that the database connection is closed, regardless of
        whether an exception occurred during the context's execution.

        Args:
            exc_type: The type of exception that caused the context to exit
                      (None if no exception occurred)
            exc_val: The exception instance that caused the context to exit
                     (None if no exception occurred)
            exc_tb: The traceback for the exception (None if no exception occurred)

        Note:
            Any exception is re-raised after the connection is closed.

        """
        self.close()

    def __del__(self) -> None:
        """
        Destructor for the Neo4j graph connection.

        This method is called during garbage collection to ensure that
        database resources are released if not explicitly closed.

        Caution:
            - Do not rely on this method for deterministic resource cleanup
            - Always prefer explicit .close() or context manager

        Best practices:
            1. Use context manager:
               with Neo4jGraph(...) as graph:
                   ...
            2. Explicitly close:
               graph = Neo4jGraph(...)
               try:
                   ...
               finally:
                   graph.close()

        """
        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass
