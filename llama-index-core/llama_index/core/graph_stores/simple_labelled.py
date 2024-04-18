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
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.schema import BaseNode, NodeWithScore
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
        embedding_dict: Optional[dict] = None,
    ) -> None:
        self.graph = graph or LabelledPropertyGraph()

    def get(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets."""
        if entity_names is None and relation_names is None and properties is None:
            return self.graph.get_triplets()

        triplets = self.graph.get_triplets()
        if entity_names:
            triplets = [
                t
                for t in triplets
                if t[0].text in entity_names or t[2].text in entity_names
            ]

        if relation_names:
            triplets = [t for t in triplets if t[1].text in relation_names]

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

        # Filter by node_ids and ref_doc_ids
        if node_ids:
            triplets = [
                t
                for t in triplets
                if any(
                    t[0].properties.get(TRIPLET_SOURCE_KEY) == i
                    or t[2].properties.get(TRIPLET_SOURCE_KEY) == i
                    for i in node_ids
                )
            ]

        return triplets

    def get_rel_map(
        self, entities: List[Entity], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triplets = []

        # depth = 0
        entity_triplets = self.get(entity_names=[entity.text for entity in entities])

        for _ in range(depth):
            for triplet in entity_triplets:
                triplets.append(triplet)
                entity_triplets = self.get(entity_names=[triplet[2].text])
                if len(entity_triplets) == 0:
                    break

        return triplets[:limit]

    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Get nodes."""
        nodes = []
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def upsert_nodes(self, nodes: List[BaseNode]) -> None:
        """Add nodes."""
        for node in nodes:
            self.graph.add_node(node)

            if node.embedding is not None:
                self.embedding_dict[node.id_] = node.embedding

    def upsert_triplets(self, triplets: List[Triplet]) -> None:
        """Add triplets."""
        for triplet in triplets:
            self.graph.add_triplet(triplet)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        triplets = self.get(
            entity_names=entity_names,
            relation_names=relation_names,
            properties=properties,
            node_ids=node_ids,
        )
        for triplet in triplets:
            self.graph.delete_triplet(triplet)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes."""
        node_ids = node_ids or []
        ref_doc_ids = ref_doc_ids or []

        # delete by node_id
        for node_id in node_ids:
            if node_id in self.graph.entities:
                del self.graph.entities[node_id]

            triplets = self.get(node_ids=[node_id])
            for triplet in triplets:
                self.graph.delete_triplet(triplet)

        # delete by ref_doc_id, which is a property of the entity
        for ref_doc_id in ref_doc_ids:
            existing_entities = list(self.graph.entities.values())
            for entity in existing_entities:
                if ref_doc_id == entity.properties.get("ref_doc_id", None):
                    del self.graph.entities[entity.text]
                    triplets = self.get(node_ids=[entity.properties.get("id_", "na")])
                    for triplet in triplets:
                        self.graph.delete_triplet(triplet)

                if ref_doc_id == entity.properties.get("id_", None):
                    del self.graph.entities[entity.text]
                    triplets = self.get(node_ids=[entity.properties.get("id_", "na")])
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

    # NOTE: Unimplemented methods for SimpleLPGStore

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        raise NotImplementedError("Schema not implemented for SimpleLPGStore.")

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> List[Entity]:
        """Query the graph store with statement and parameters."""
        raise NotImplementedError(
            "Structured query not implemented for SimpleLPGStore."
        )

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> List[NodeWithScore]:
        """Query the graph store with a vector store query."""
        raise NotImplementedError("Vector query not implemented for SimpleLPGStore.")

    @property
    def client(self) -> Any:
        """Get client."""
        raise NotImplementedError("Client not implemented for SimpleLPGStore.")
