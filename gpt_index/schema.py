"""Base schema for data structures."""
from dataclasses import dataclass
from typing import Dict, Optional, Set

from dataclasses_json import DataClassJsonMixin


@dataclass
class Document:
    """Generic interface for document."""

    text: str
    extra_info: Optional[Dict] = None


@dataclass
class Node(DataClassJsonMixin):
    """A node in the GPT tree index."""

    text: str
    index: int
    child_indices: Set[int]


@dataclass
class IndexGraph(DataClassJsonMixin):
    """A graph representing the tree-structured index."""

    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]

    @property
    def size(self) -> int:
        """Get the size of the graph."""
        return len(self.all_nodes)
