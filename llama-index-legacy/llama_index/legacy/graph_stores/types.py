from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import fsspec

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"


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
