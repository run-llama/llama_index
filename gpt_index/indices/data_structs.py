"""File for core data structures."""

from dataclasses import dataclass
from typing import Dict, Set

from dataclasses_json import DataClassJsonMixin


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a GPT index."""


@dataclass
class Node(DataClassJsonMixin):
    """A node in the GPT tree index."""

    text: str
    index: int
    child_indices: Set[int]


@dataclass
class IndexGraph(IndexStruct):
    """A graph representing the tree-structured index."""

    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]

    @property
    def size(self) -> int:
        """Get the size of the graph."""
        return len(self.all_nodes)
