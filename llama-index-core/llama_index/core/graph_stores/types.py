from typing import Any, Dict, List, Optional, Tuple, Set, Protocol, runtime_checkable

import fsspec

from llama_index.core.bridge.pydantic import BaseModel, Field

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"
DEFUALT_LPG_PERSIST_FNAME = "lpg_graph_store.json"


class Entity(BaseModel):
    """An entity in a graph."""

    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A relation connecting two entities in a graph."""

    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)


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
        if (subj.name, rel.name, obj.name) in self.triplets:
            return

        self.triplets.add((subj.name, rel.name, obj.name))
        self.entities[subj.name] = subj
        self.entities[obj.name] = obj
        self.relations[rel.name] = rel

    def delete_triplet(self, triplet: Triplet) -> None:
        """Delete a triplet."""
        subj, rel, obj = triplet
        if (subj.name, rel.name, obj.name) not in self.triplets:
            return

        self.triplets.remove((subj.name, rel.name, obj.name))
        if subj.name in self.entities:
            del self.entities[subj.name]
        if obj.name in self.entities:
            del self.entities[obj.name]
        if rel.name in self.relations:
            del self.relations[rel.name]


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
    ) -> List[Triplet]:
        """Get triplets."""
        ...

    def get_rel_map(
        self, entities: List[Entity], depth: int = 2, limit: int = 30
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        ...

    def upsert_triplet(self, triplets: List[Triplet]) -> None:
        """Add triplets."""
        ...

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
    ) -> None:
        """Delete matching data."""
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        return

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        ...

    def query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> List[Triplet]:
        """Query the graph store with statement and parameters."""
        ...
