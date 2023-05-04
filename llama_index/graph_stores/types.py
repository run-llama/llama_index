from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import fsspec

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"


@runtime_checkable
class GraphStore(Protocol):
    """Abstract graph store protocol."""

    @property
    def client(self) -> Any:
        ...

    def get(self, sub: str) -> List[List[str]]:
        ...

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        ...

    def upsert_triplet(self, sub: str, rel: str, obj: str) -> None:
        ...

    def delete(self, sub: str, rel: str, obj: str) -> None:
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        return None
