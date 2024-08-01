"""Kùzu graph store index."""

from typing import Any, Dict, List, Optional

from llama_index.legacy.graph_stores.types import GraphStore


class KuzuGraphStore(GraphStore):
    def __init__(
        self,
        database: Any,
        node_table_name: str = "entity",
        rel_table_name: str = "links",
        **kwargs: Any,
    ) -> None:
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        self.database = database
        self.connection = kuzu.Connection(database)
        self.node_table_name = node_table_name
        self.rel_table_name = rel_table_name
        self.init_schema()

    def init_schema(self) -> None:
        """Initialize schema if the tables do not exist."""
        node_tables = self.connection._get_node_table_names()
        if self.node_table_name not in node_tables:
            self.connection.execute(
                "CREATE NODE TABLE %s (ID STRING, PRIMARY KEY(ID))"
                % self.node_table_name
            )
        rel_tables = self.connection._get_rel_table_names()
        rel_tables = [rel_table["name"] for rel_table in rel_tables]
        if self.rel_table_name not in rel_tables:
            self.connection.execute(
                "CREATE REL TABLE {} (FROM {} TO {}, predicate STRING)".format(
                    self.rel_table_name, self.node_table_name, self.node_table_name
                )
            )

    @property
    def client(self) -> Any:
        return self.connection

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r:%s]->(n2:%s)
            WHERE n1.ID = $subj
            RETURN r.predicate, n2.ID;
        """
        prepared_statement = self.connection.prepare(
            query % (self.node_table_name, self.rel_table_name, self.node_table_name)
        )
        query_result = self.connection.execute(prepared_statement, [("subj", subj)])
        retval = []
        while query_result.has_next():
            row = query_result.get_next()
            retval.append([row[0], row[1]])
        return retval

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        rel_wildcard = "r:%s*1..%d" % (self.rel_table_name, depth)
        match_clause = "MATCH (n1:{})-[{}]->(n2:{})".format(
            self.node_table_name,
            rel_wildcard,
            self.node_table_name,
        )
        return_clause = "RETURN n1, r, n2 LIMIT %d" % limit
        params = []
        if subjs is not None:
            for i, curr_subj in enumerate(subjs):
                if i == 0:
                    where_clause = "WHERE n1.ID = $%d" % i
                else:
                    where_clause += " OR n1.ID = $%d" % i
                params.append((str(i), curr_subj))
        else:
            where_clause = ""
        query = f"{match_clause} {where_clause} {return_clause}"
        prepared_statement = self.connection.prepare(query)
        if subjs is not None:
            query_result = self.connection.execute(prepared_statement, params)
        else:
            query_result = self.connection.execute(prepared_statement)
        retval: Dict[str, List[List[str]]] = {}
        while query_result.has_next():
            row = query_result.get_next()
            curr_path = []
            subj = row[0]
            recursive_rel = row[1]
            obj = row[2]
            nodes_map = {}
            nodes_map[(subj["_id"]["table"], subj["_id"]["offset"])] = subj["ID"]
            nodes_map[(obj["_id"]["table"], obj["_id"]["offset"])] = obj["ID"]
            for node in recursive_rel["_nodes"]:
                nodes_map[(node["_id"]["table"], node["_id"]["offset"])] = node["ID"]
            for rel in recursive_rel["_rels"]:
                predicate = rel["predicate"]
                curr_subj_id = nodes_map[(rel["_src"]["table"], rel["_src"]["offset"])]
                curr_path.append(curr_subj_id)
                curr_path.append(predicate)
            # Add the last node
            curr_path.append(obj["ID"])
            if subj["ID"] not in retval:
                retval[subj["ID"]] = []
            retval[subj["ID"]].append(curr_path)
        return retval

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""

        def check_entity_exists(connection: Any, entity: str) -> bool:
            is_exists_result = connection.execute(
                "MATCH (n:%s) WHERE n.ID = $entity RETURN n.ID" % self.node_table_name,
                [("entity", entity)],
            )
            return is_exists_result.has_next()

        def create_entity(connection: Any, entity: str) -> None:
            connection.execute(
                "CREATE (n:%s {ID: $entity})" % self.node_table_name,
                [("entity", entity)],
            )

        def check_rel_exists(connection: Any, subj: str, obj: str, rel: str) -> bool:
            is_exists_result = connection.execute(
                (
                    "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.ID = $subj AND n2.ID = "
                    "$obj AND r.predicate = $pred RETURN r.predicate"
                ).format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                [("subj", subj), ("obj", obj), ("pred", rel)],
            )
            return is_exists_result.has_next()

        def create_rel(connection: Any, subj: str, obj: str, rel: str) -> None:
            connection.execute(
                (
                    "MATCH (n1:{}), (n2:{}) WHERE n1.ID = $subj AND n2.ID = $obj "
                    "CREATE (n1)-[r:{} {{predicate: $pred}}]->(n2)"
                ).format(
                    self.node_table_name, self.node_table_name, self.rel_table_name
                ),
                [("subj", subj), ("obj", obj), ("pred", rel)],
            )

        is_subj_exists = check_entity_exists(self.connection, subj)
        is_obj_exists = check_entity_exists(self.connection, obj)

        if not is_subj_exists:
            create_entity(self.connection, subj)
        if not is_obj_exists:
            create_entity(self.connection, obj)

        if is_subj_exists and is_obj_exists:
            is_rel_exists = check_rel_exists(self.connection, subj, obj, rel)
            if is_rel_exists:
                return

        create_rel(self.connection, subj, obj, rel)

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(connection: Any, subj: str, obj: str, rel: str) -> None:
            connection.execute(
                (
                    "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.ID = $subj AND n2.ID"
                    " = $obj AND r.predicate = $pred DELETE r"
                ).format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                [("subj", subj), ("obj", obj), ("pred", rel)],
            )

        def delete_entity(connection: Any, entity: str) -> None:
            connection.execute(
                "MATCH (n:%s) WHERE n.ID = $entity DELETE n" % self.node_table_name,
                [("entity", entity)],
            )

        def check_edges(connection: Any, entity: str) -> bool:
            is_exists_result = connection.execute(
                "MATCH (n1:{})-[r:{}]-(n2:{}) WHERE n2.ID = $entity RETURN r.predicate".format(
                    self.node_table_name, self.rel_table_name, self.node_table_name
                ),
                [("entity", entity)],
            )
            return is_exists_result.has_next()

        delete_rel(self.connection, subj, obj, rel)
        if not check_edges(self.connection, subj):
            delete_entity(self.connection, subj)
        if not check_edges(self.connection, obj):
            delete_entity(self.connection, obj)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        node_table_name: str = "entity",
        rel_table_name: str = "links",
    ) -> "KuzuGraphStore":
        """Load from persist dir."""
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        database = kuzu.Database(persist_dir)
        return cls(database, node_table_name, rel_table_name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KuzuGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)
