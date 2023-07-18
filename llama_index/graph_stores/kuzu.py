"""Kùzu graph store index."""
import logging
import os
from string import Template
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from llama_index.graph_stores.types import GraphStore


class KuzuGraphStore(GraphStore):
    def __init__(self, database, node_table_name="entity", rel_table_name="links", **kwargs):
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        self.database = database
        self.connection = kuzu.Connection(database)
        self.node_table_name = node_table_name
        self.rel_table_name = rel_table_name
        self.init_schema()

    def init_schema(self):
        node_tables = self.connection._get_node_table_names()
        if self.node_table_name not in node_tables:
            self.connection.execute(
                "CREATE NODE TABLE %s (ID STRING, PRIMARY KEY(ID))" % self.node_table_name
            )
        rel_tables = self.connection._get_rel_table_names()
        rel_tables = [rel_table["name"] for rel_table in rel_tables]
        if self.rel_table_name not in rel_tables:
            self.connection.execute(
                "CREATE REL TABLE %s (FROM %s TO %s, predicate STRING)" % (
                    self.rel_table_name, self.node_table_name, self.node_table_name)
            )

    @property
    def client(self):
        return self.connection

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r:%s]->(n2:%s)
            WHERE n1.ID = $subj
            RETURN r.predicate, n2.ID;
        """
        prepared_statement = self.connection.prepare(
            query % (self.node_table_name, self.rel_table_name, self.node_table_name))
        query_result = self.connection.execute(
            prepared_statement, [("subj", subj)])
        retval = []
        while query_result.has_next():
            row = query_result.get_next()
            retval.append([row[0], row[1]])
        return retval

    def get_rel_map(
            self, sinjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        def check_entity_exists(connection, entity):
            is_exists_result = connection.execute(
                "MATCH (n:%s) WHERE n.ID = $entity RETURN n.ID" % self.node_table_name,
                [("entity", entity)]
            )
            return is_exists_result.has_next()

        def create_entity(connection, entity):
            connection.execute(
                "CREATE (n:%s {ID: $entity})" % self.node_table_name,
                [("entity", entity)]
            )

        def check_rel_exists(connection, subj, obj, rel):
            is_exists_result = connection.execute(
                "MATCH (n1:%s)-[r:%s]->(n2:%s) WHERE n1.ID = $subj AND n2.ID = $obj AND r.predicate = $pred RETURN r.predicate" % (
                    self.node_table_name, self.rel_table_name, self.node_table_name),
                [("subj", subj), ("obj", obj), ("pred", rel)]
            )
            return is_exists_result.has_next()

        def create_rel(connection, subj, obj, rel):
            connection.execute(
                "MATCH (n1:%s), (n2:%s) WHERE n1.ID = $subj AND n2.ID = $obj CREATE (n1)-[r:%s {predicate: $pred}]->(n2)" % (
                    self.node_table_name, self.node_table_name, self.rel_table_name),
                [("subj", subj), ("obj", obj), ("pred", rel)]
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
        def delete_rel(connection, subj, obj, rel):
            connection.execute(
                "MATCH (n1:%s)-[r:%s]->(n2:%s) WHERE n1.ID = $subj AND n2.ID = $obj AND r.predicate = $pred DELETE r" % (
                    self.node_table_name, self.rel_table_name, self.node_table_name),
                [("subj", subj), ("obj", obj), ("pred", rel)])

        def delete_entity(connection, entity):
            connection.execute(
                "MATCH (n:%s) WHERE n.ID = $entity DELETE n" % self.node_table_name,
                [("entity", entity)]
            )

        def check_edges(connection, entity):
            is_exists_result = connection.execute(
                "MATCH (n1:%s)-[r:%s]-(n2:%s) WHERE n2.ID = $entity RETURN r.predicate" % (
                    self.node_table_name, self.rel_table_name, self.node_table_name),
                [("entity", entity)]
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
            node_table_name="entity", rel_table_name="links"
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
