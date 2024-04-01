import fsspec
import os
from typing import Any, List, Dict, Optional

from llama_index.core.graph_stores.types import (
    LabelledPropertyGraphStore,
    Triplet,
    Entity,
    LabelledPropertyGraph,
    DEFAULT_PERSIST_DIR,
    DEFUALT_LPG_PERSIST_FNAME,
)
from llama_index.core.schema import BaseNode


class SimpleLPGStore(LabelledPropertyGraphStore):
    """Simple Labelled Property Graph Store.

    This class implements a simple in-memory labelled property graph store.

    Args:
        graph (Optional[LabelledPropertyGraph]): Labelled property graph to initialize the store.
    """

    supports_vectors: bool = False
    supports_queries: bool = False
    supports_nodes: bool = False

    def __init__(self, graph: Optional[LabelledPropertyGraph] = None):
        self.graph = graph or LabelledPropertyGraph()

    @property
    def client(self) -> Any:
        """Get client."""
        raise NotImplementedError("Client not implemented for SimpleLPGStore.")

    def get(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
    ) -> List[Triplet]:
        """Get triplets."""
        if entity_names is None and relation_names is None and properties is None:
            return self.graph.get_triplets()

        triplets = self.graph.get_triplets()
        if entity_names:
            triplets = [
                t
                for t in triplets
                if t[0].name in entity_names or t[2].name in entity_names
            ]

        if relation_names:
            triplets = [t for t in triplets if t[1].name in relation_names]

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

        return triplets

    def get_by_ids(self, node_ids: List[str] = None) -> List[BaseNode]:
        raise NotImplementedError("Get by ids not implemented for SimpleLPGStore.")

    def get_rel_map(
        self, entities: List[Entity], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triplets = []

        # depth = 0
        entity_triplets = self.get(entity_names=[entity.name for entity in entities])

        for _ in range(depth):
            for triplet in entity_triplets:
                triplets.append(triplet)
                entity_triplets = self.get(entity_names=[triplet[2].name])
                if len(entity_triplets) == 0:
                    break

        return triplets[:limit]

    def upsert_triplets(self, triplets: List[Triplet]) -> None:
        """Add triplets."""
        for triplet in triplets:
            self.graph.add_triplet(triplet)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
    ) -> None:
        """Delete matching data."""
        triplets = self.get(
            entity_names=entity_names,
            relation_names=relation_names,
            properties=properties,
        )
        for triplet in triplets:
            self.graph.delete_triplet(triplet)

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

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        raise NotImplementedError("Schema not implemented for SimpleLPGStore.")

    def query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> List[Triplet]:
        """Query the graph store with statement and parameters."""
        raise NotImplementedError("Query not implemented for SimpleLPGStore.")
