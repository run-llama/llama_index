from abc import abstractmethod
import logging
from typing import Any, Dict, List, Optional
from llama_index.core.graph_stores.types import GraphStore
from .neptune import refresh_schema

logger = logging.getLogger(__name__)


class NeptuneBaseGraphStore(GraphStore):
    """This is an abstract base class that represents the shared features across the NeptuneDatabaseGraphStore
    and NeptuneAnalyticsGraphStore classes.
    """

    def __init__() -> None:
        pass

    @property
    def client(self) -> Any:
        return self._client

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
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            return rel_map

        query = f"""MATCH p=(n1:{self.node_label})-[*1..{depth}]->() WHERE n1.id IN $subjs
            UNWIND relationships(p) AS rel WITH n1.id AS subj, p,
            collect([type(rel), endNode(rel).id])AS flattened_rels
            UNWIND flattened_rels as fr
            WITH DISTINCT fr, subj
            RETURN subj, collect(fr) AS flattened_rels LIMIT {limit}"""

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            rel_map[record["subj"]] = record["flattened_rels"]
        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet to the graph."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[:`%s`]->(n2)
        """

        prepared_statement = query % (
            self.node_label.replace("`", ""),
            rel.replace(" ", "_").replace("`", "").upper(),
        )

        self.query(
            prepared_statement,
            {"subj": subj.replace("`", ""), "obj": obj.replace("`", "")},
        )

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet from the graph."""

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
                    "MATCH (n:%s) WHERE n.id = $entity DETACH DELETE n"
                    % self.node_label,
                    {"entity": entity},
                )

        delete_rel(subj, obj, rel)
        delete_entity(subj)

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the Neptune KG store."""
        if refresh or not self.schema:
            self.schema = refresh_schema(self.query, self._get_summary())["schema_str"]

        return self.schema

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_summary(self) -> Dict:
        raise NotImplementedError
