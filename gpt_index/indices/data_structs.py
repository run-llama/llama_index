"""File for core data structures."""

import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set

from dataclasses_json import DataClassJsonMixin


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a GPT index."""


@dataclass
class Node(IndexStruct):
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


@dataclass
class KeywordTable(IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    table: Dict[str, Set[int]] = field(default_factory=dict)
    text_chunks: Dict[int, str] = field(default_factory=dict)

    def _get_index(self) -> int:
        """Get the next index for the text chunk."""
        # randomly generate until we get a unique index
        while True:
            idx = random.randint(0, sys.maxsize)
            if idx not in self.text_chunks:
                break
        return idx

    def add_text(self, keywords: List[str], text_chunk: str) -> int:
        """Add text to table."""
        cur_idx = self._get_index()
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(cur_idx)
        self.text_chunks[cur_idx] = text_chunk
        return cur_idx

    def get_texts(self, keyword: str) -> List[str]:
        """Get texts given keyword."""
        if keyword not in self.table:
            raise ValueError("Keyword not found in table.")
        return [self.text_chunks[idx] for idx in self.table[keyword]]

    @property
    def keywords(self) -> Set[str]:
        """Get all keywords in the table."""
        return set(self.table.keys())

    @property
    def size(self) -> int:
        """Get the size of the table."""
        return len(self.table)


@dataclass
class IndexList(IndexStruct):
    """A list of documents."""

    nodes: List[Node] = field(default_factory=list)

    def add_text(self, text_chunk: str) -> int:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        cur_node = Node(text=text_chunk, index=len(self.nodes), child_indices=set())
        self.nodes.append(cur_node)
        return cur_node.index
