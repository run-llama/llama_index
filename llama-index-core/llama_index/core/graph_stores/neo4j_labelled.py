from typing import Any, List, Dict, Optional
import neo4j

from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    Relation,
)


class Neo4jLPGStore(PropertyGraphStore):
    """Simple Labelled Property Graph Store.

    This class implements a simple in-memory labelled property graph store.

    Args:
        graph (Optional[LabelledPropertyGraph]): Labelled property graph to initialize the store.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
    ) -> None:
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        nodes = list(self.graph.nodes.values())
        if properties:
            nodes = [
                n
                for n in nodes
                if any(n.properties.get(k) == v for k, v in properties.items())
            ]

        # Filter by node_ids
        if ids:
            nodes = [n for n in nodes if n.id in ids]

        return nodes

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        params = []
        print(nodes)
        # for node in nodes:
        ##    if node.name: # entity node
        #       label
        #       embedding
        #       properties
        #   else: #vector node

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        print(relations)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        triplets = self.get_triplets(
            entity_names=entity_names,
            relation_names=relation_names,
            properties=properties,
            ids=ids,
        )
        for triplet in triplets:
            self.graph.delete_triplet(triplet)

        nodes = self.get(properties=properties, ids=ids)
        for node in nodes:
            self.graph.delete_node(node)

    def database_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> Any:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]

    def get_rel_map(self):
        pass

    def get_schema(self):
        pass

    def get_triplets(self):
        pass

    def persist(self):
        pass

    def structured_query(self):
        pass

    def upsert_triplets(self):
        pass

    def vector_query(self):
        pass
