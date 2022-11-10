"""Base schema for data structures."""
from dataclasses import dataclass
from typing import Dict, Set

from dataclasses_json import DataClassJsonMixin


@dataclass
class Node(DataClassJsonMixin):
    """A node in the GPT index."""

    text: str
    index: int
    child_indices: Set[int]


@dataclass
class IndexGraph(DataClassJsonMixin):
    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]

    @property
    def size(self):
        return len(self.all_nodes)
