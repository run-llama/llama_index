import fsspec
import os
from typing import Any, List, Dict, Tuple, Optional

from llama_index.core.graph_stores.types import (
    LabelledPropertyGraphStore,
    Triplet,
    LabelledNode,
    LabelledPropertyGraph,
    Relation,
    DEFAULT_PERSIST_DIR,
    DEFUALT_LPG_PERSIST_FNAME,
)
from llama_index.core.vector_stores.types import VectorStoreQuery


class SimpleLPGStore(LabelledPropertyGraphStore):
    """Simple Labelled Property Graph Store.

    This class implements a simple in-memory labelled property graph store.

    Args:
        graph (Optional[LabelledPropertyGraph]): Labelled property graph to initialize the store.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False

    def __init__(
        self,
        graph: Optional[LabelledPropertyGraph] = None,
    ) -> None:
        self.graph = graph or LabelledPropertyGraph()

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

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets."""
        # if nothing is passed, return empty list
        if not ids and not properties and not entity_names and not relation_names:
            return []

        triplets = self.graph.get_triplets()
        if entity_names:
            triplets = [
                t
                for t in triplets
                if t[0].id in entity_names or t[2].id in entity_names
            ]

        if relation_names:
            triplets = [t for t in triplets if t[1].id in relation_names]

        if properties:
            triplets = [
                t
                for t in triplets
                if any(
                    t[0].properties.get(k) == v
                    or t[1].properties.get(k) == v
                    or t[2].properties.get(k) == v
                    for k, v in properties.items()
                )
            ]

        # Filter by node_ids
        if ids:
            triplets = [
                t for t in triplets if any(t[0].id == i or t[2].id == i for i in ids)
            ]

        return triplets

    def get_rel_map(
        self, graph_nodes: List[LabelledNode], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triplets = []

        cur_depth = 0
        graph_triplets = self.get_triplets(ids=[gn.id for gn in graph_nodes])
        seen_triplets = set()

        while len(graph_triplets) > 0 and cur_depth < depth:
            triplets.extend(graph_triplets)

            # get next depth
            graph_triplets = self.get_triplets(
                entity_names=[t[2].id for t in graph_triplets]
            )
            graph_triplets = [t for t in graph_triplets if str(t) not in seen_triplets]
            seen_triplets.update([str(t) for t in graph_triplets])
            depth += 1

        return triplets[:limit]

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """Add nodes."""
        for node in nodes:
            self.graph.add_node(node)

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        for relation in relations:
            self.graph.add_relation(relation)

    def upsert_triplets(self, triplets: List[Triplet]) -> None:
        """Add triplets."""
        for triplet in triplets:
            self.graph.add_triplet(triplet)

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

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        if fs is None:
            fs = fsspec.filesystem("file")
        with fs.open(persist_path, "w") as f:
            f.write(self.graph.json())

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleLPGStore":
        """Load from persist path."""
        if fs is None:
            fs = fsspec.filesystem("file")
        with fs.open(persist_path, "r") as f:
            graph = LabelledPropertyGraph.parse_raw(f.read())
        return cls(graph)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleLPGStore":
        """Load from persist dir."""
        persist_path = os.path.join(persist_dir, DEFUALT_LPG_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, fs=fs)

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "SimpleLPGStore":
        """Load from dict."""
        graph = LabelledPropertyGraph.parse_obj(data)
        return cls(graph)

    def to_dict(self) -> dict:
        """Convert to dict."""
        return self.graph.dict()

    # NOTE: Unimplemented methods for SimpleLPGStore

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        raise NotImplementedError("Schema not implemented for SimpleLPGStore.")

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query the graph store with statement and parameters."""
        raise NotImplementedError(
            "Structured query not implemented for SimpleLPGStore."
        )

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> List[Tuple[LabelledNode, float]]:
        """Query the graph store with a vector store query."""
        raise NotImplementedError("Vector query not implemented for SimpleLPGStore.")

    @property
    def client(self) -> Any:
        """Get client."""
        raise NotImplementedError("Client not implemented for SimpleLPGStore.")

    def save_networkx_graph(self) -> None:
        """Display the graph store."""
        import networkx as nx

        G = nx.DiGraph()
        for node in self.graph.nodes.values():
            G.add_node(node.id, label=node.id)
        for triplet in self.graph.triplets:
            G.add_edge(triplet[0], triplet[2], label=triplet[1])

        # save to html file
        from pyvis.network import Network

        net = Network(notebook=False, directed=True)
        net.from_nx(G)
        net.write_html("kg.html")
