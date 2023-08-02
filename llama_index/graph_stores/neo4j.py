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
            with self._driver.session() as session:
                session.run(
                    """
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:%s) REQUIRE n.id IS UNIQUE; 
                """
                    % (self.node_label)
                )
        except:  # Using Neo4j <5
            with self._driver.session() as session:
                session.run(
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
        
        with self._driver.session() as session:
            data = session.run(prepared_statement, {"subj": subj})

        retval = [record.value() for record in data]
        return retval
    
    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
     ) -> Dict[str, List[List[str]]]:
        return {"test": [["123", "123"]]}

    # def get_rel_map(
    #    self, subjs: Optional[List[str]] = None, depth: int = 2
    # ) -> Dict[str, List[List[str]]]:
    #    """Get depth-aware rel map."""
    #    rel_wildcard = "r:%s*1..%d" % (self.rel_table_name, depth)
    #    match_clause = "MATCH (n1:%s)-[%s]->(n2:%s)" % (
    #        self.node_label,
    #        rel_wildcard,
    #        self.node_label,
    #    )
    #    return_clause = "RETURN n1, r, n2"
    #    params = []
    #    if subjs is not None:
    #        for i, curr_subj in enumerate(subjs):
    #            if i == 0:
    #                where_clause = "WHERE n1.ID = $%d" % i
    #            else:
    #                where_clause += " OR n1.ID = $%d" % i
    #            params.append((str(i), curr_subj))
    #    else:
    #        where_clause = ""
    #    query = "%s %s %s" % (match_clause, where_clause, return_clause)
    #    prepared_statement = self.connection.prepare(query)
    #    if subjs is not None:
    #        query_result = self.connection.execute(prepared_statement, params)
    #    else:
    #        query_result = self.connection.execute(prepared_statement)
    #    retval: Dict[str, List[List[str]]] = {}
    #    while query_result.has_next():
    #        row = query_result.get_next()
    #        curr_path = []
    #        subj = row[0]
    #        recursive_rel = row[1]
    #        obj = row[2]
    #        nodes_map = {}
    #        nodes_map[(subj["_id"]["table"], subj["_id"]["offset"])] = subj["ID"]
    #        nodes_map[(obj["_id"]["table"], obj["_id"]["offset"])] = obj["ID"]
    #        for node in recursive_rel["_nodes"]:
    #            nodes_map[(node["_id"]["table"], node["_id"]["offset"])] = node["ID"]
    #        for rel in recursive_rel["_rels"]:
    #            predicate = rel["predicate"]
    #            curr_subj_id = nodes_map[(rel["_src"]["table"], rel["_src"]["offset"])]
    #            curr_path.append(curr_subj_id)
    #            curr_path.append(predicate)
    #        # Add the last node
    #        curr_path.append(obj["ID"])
    #        # Remove subject as it is the key of the map
    #        curr_path = curr_path[1:]
    #        if subj["ID"] not in retval:
    #            retval[subj["ID"]] = []
    #        retval[subj["ID"]].append(curr_path)
    #    return retval

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""

        query = """
            MERGE (n1:%s {id:$subj})
            MERGE (n2:%s {id:$obj})
            MERGE (n1)-[:%s]->(n2)
        """

        prepared_statement = query % (self.node_label, rel, self.node_label)

        with self._driver.session() as session:
            session.run(prepared_statement)

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(connection: Any, subj: str, obj: str, rel: str) -> None:
            with self._driver.session() as session:
                connection.run(
                    (
                        "MATCH (n1:%s)-[r:%s]->(n2:%s) WHERE n1.id = $subj AND n2.id"
                        " = $obj AND r.predicate = $pred DELETE r"
                    )
                    % (self.node_label, rel, self.node_label),
                    {"subj": subj, "obj": obj},
                )

        def delete_entity(connection: Any, entity: str) -> None:
            with self._driver.session() as session:
                connection.run(
                    "MATCH (n:%s) WHERE n.id = $entity DELETE n" % self.node_label,
                    {"entity": entity},
                )

        def check_edges(connection: Any, entity: str) -> bool:
            is_exists_result = connection.execute(
                "MATCH (n1:%s) WHERE n1.id = $entity RETURN count(*)"
                % (self.node_label),
                {"entity": entity},
            )
            return bool(list(is_exists_result))

        delete_rel(self._driver, subj, obj, rel)
        if not check_edges(self._driver, subj):
            delete_entity(self._driver, subj)
        if not check_edges(self._driver, obj):
            delete_entity(self._driver, obj)

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

        with self._driver.session() as session:
            result = session.run(query, param_map)

        return [d.data() for d in result]
