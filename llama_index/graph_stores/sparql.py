from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging
import fsspec

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"

logging.basicConfig(filename='loggy.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('sparql HERE')


@runtime_checkable
class SparqlStore(GraphStore):
    """SPARQL graph store

    #### TODO This protocol defines the interface for a graph store, which is responsible
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
        logger.info('#### sparql client(self) called')
        ...

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        logger.info('#### sparql get called')
        ...

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        logger.info('#### sparql get_rel_map called')
        ...

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        logger.info('#### sparql upsert_triplet called')
        ...

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        logger.info('#### sparql delete called')
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        logger.info('#### sparql persist called')
        return None

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        logger.info('#### get_schema called')
        ...

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        logger.info('#### sparql query called')
        ...
