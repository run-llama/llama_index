"""File for core data structures."""

import random
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from dataclasses_json import DataClassJsonMixin

from gpt_index.schema import Document, DocumentStore


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a GPT index."""

    doc_id: Optional[str]

    def _create_document(doc: Optional[Document] = None) -> Document:
        """Create document.

        This method is used to create a document from the index struct, which
        will be registered in the document store. This method
        should not be called directly.

        """
        raise NotImplementedError("Not iplemented yet.")

    def register_doc(
        self,
        doc_store: DocumentStore,
        doc_id: Optional[str] = None,
        doc: Optional[Document] = None,
    ) -> None:
        """Register in document store.

        This registers a document_id for the index struct. This is useful for
        being able to construct higher-level indices that are based on lower-level
        indices, since each index struct maps to a given document_id.

        In order for a subclass to register a document_id, the subclass must
        create a text document through _create_document.
        This text can either be passed in as an optional argument,
        or it can be synthesized from the index struct itself.

        """
        doc_id = doc_store.get_new_id() if doc_id is None else doc_id
        doc = self._create_document(doc=doc)
        doc_store = doc_store.add_document(doc)

    def get_text(self, doc_store: DocumentStore) -> str:
        """Get the text of the index struct."""
        if self.doc_id is None:
            raise ValueError("self.doc_id must not be None.")
        return doc_store.get_document(self.doc_id).text


@dataclass
class Node(IndexStruct):
    """A generic node of data.

    Used in the GPT Tree Index and List Index.

    """

    # used for GPTTreeIndex
    child_indices: Set[int] = field(default_factory=set)


@dataclass
class IndexGraph(IndexStruct):
    """A graph representing the tree-structured index."""

    all_nodes: Dict[int, Node] = field(default_factory=dict)
    root_nodes: Dict[int, Node] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get the size of the graph."""
        return len(self.all_nodes)

    def get_children(self, parent_node: Optional[Node]) -> Dict[int, Node]:
        """Get nodes given indices."""
        if parent_node is None:
            return self.root_nodes
        else:
            return {i: self.all_nodes[i] for i in parent_node.child_indices}

    def insert_under_parent(self, node: Node, parent_node: Optional[Node]) -> None:
        """Insert under parent node."""
        if node.index in self.all_nodes:
            raise ValueError(
                "Cannot insert a new node with the same index as an existing node."
            )
        if parent_node is None:
            self.root_nodes[node.index] = node
        else:
            parent_node.child_indices.add(node.index)

        self.all_nodes[node.index] = node


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
