from typing import Any, Dict, List, Optional, Tuple, Set, Protocol, runtime_checkable

import fsspec

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"
DEFUALT_LPG_PERSIST_FNAME = "lpg_graph_store.json"

TRIPLET_SOURCE_KEY = "triplet_source_id"


class Entity(BaseModel):
    """An entity in a graph."""

    text: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class Relation(BaseModel):
    """A relation connecting two entities in a graph."""

    text: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


Triplet = Tuple[Entity, Relation, Entity]


class LabelledPropertyGraph(BaseModel):
    """In memory labelled property graph containing entities and relations."""

    entities: Dict[str, Entity] = Field(default_factory=dict)
    relations: Dict[str, Relation] = Field(default_factory=dict)
    triplets: Set[Tuple[str, str, str]] = Field(
        default_factory=set, description="List of triplets (subject, relation, object)."
    )

    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return list(self.entities.values())

    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        return list(self.relations.values())

    def get_triplets(self) -> List[Triplet]:
        """Get all triplets."""
        return [
            (self.entities[subj], self.relations[rel], self.entities[obj])
            for subj, rel, obj in self.triplets
        ]

    def add_triplet(self, triplet: Triplet) -> None:
        """Add a triplet."""
        subj, rel, obj = triplet
        if (subj.text, rel.text, obj.text) in self.triplets:
            return

        self.triplets.add((subj.text, rel.text, obj.text))
        self.entities[subj.text] = subj
        self.entities[obj.text] = obj
        self.relations[rel.text] = rel

    def add_node(self, node: BaseNode) -> None:
        """Add a node."""
        metadata_dict = node_to_metadata_dict(node)
        metadata_dict["id_"] = node.id_
        self.entities[node.id_] = Entity(text=node.id_, properties=metadata_dict)

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get a node."""
        if node_id in self.entities:
            return metadata_dict_to_node(self.entities[node_id].properties)
        return None

    def delete_triplet(self, triplet: Triplet) -> None:
        """Delete a triplet."""
        subj, rel, obj = triplet
        if (subj.text, rel.text, obj.text) not in self.triplets:
            return

        self.triplets.remove((subj.text, rel.text, obj.text))
        if subj.text in self.entities:
            del self.entities[subj.text]
        if obj.text in self.entities:
            del self.entities[obj.text]
        if rel.text in self.relations:
            del self.relations[rel.text]


@runtime_checkable
class GraphStore(Protocol):
    """Abstract graph store protocol.

    This protocol defines the interface for a graph store, which is responsible
    for storing and retrieving knowledge graph data.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
        get_schema: Callable[[bool], str]: Get the schema of the graph store.
    """

    schema: str = ""

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        ...

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        ...

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        ...

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        return

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        ...

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        ...


@runtime_checkable
class LabelledPropertyGraphStore(Protocol):
    """Abstract labelled graph store protocol.

    This protocol defines the interface for a graph store, which is responsible
    for storing and retrieving knowledge graph data.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
        get_schema: Callable[[bool], str]: Get the schema of the graph store.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False
    schema: str = ""

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    def get(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets with matching values."""
        ...

    async def aget(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Asynchronously get triplets with matching values."""
        return self.get(entity_names, relation_names, properties, node_ids)

    def get_rel_map(
        self, entities: List[Entity], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        ...

    async def aget_rel_map(
        self, entities: List[Entity], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Asynchronously get depth-aware rel map."""
        return self.get_rel_map(entities, depth, limit)

    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Get nodes."""
        ...

    async def aget_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Asynchronously get nodes."""
        return self.get_nodes(node_ids)

    def upsert_nodes(self, nodes: List[BaseNode]) -> None:
        """Add nodes."""
        ...

    async def aupsert_nodes(self, nodes: List[BaseNode]) -> None:
        """Asynchronously add nodes."""
        return self.upsert_nodes(nodes)

    def upsert_triplets(self, triplets: List[Triplet]) -> None:
        """Upsert triplets."""
        ...

    async def aupsert_triplets(self, triplets: List[Triplet]) -> None:
        """Asynchronously upsert triplets."""
        return self.upsert_triplets(triplets)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        ...

    async def adelete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Asynchronously delete matching data."""
        return self.delete(entity_names, relation_names, properties, node_ids)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes."""
        ...

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Asynchronously delete nodes."""
        return self.delete_nodes(node_ids, ref_doc_ids)

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> List[Entity]:
        """Query the graph store with statement and parameters."""
        ...

    async def astructured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> List[Entity]:
        """Asynchronously query the graph store with statement and parameters."""
        return self.structured_query(query, param_map)

    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> List[Entity]:
        """Query the graph store with a vector store query."""
        ...

    async def avector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> List[Entity]:
        """Asynchronously query the graph store with a vector store query."""
        return self.vector_query(query, **kwargs)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        return

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        ...

    async def aget_schema(self, refresh: bool = False) -> str:
        """Asynchronously get the schema of the graph store."""
        return self.get_schema(refresh=refresh)
