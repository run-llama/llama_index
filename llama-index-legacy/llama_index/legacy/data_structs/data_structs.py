"""Data structures.

Nodes are decoupled from the indices.

"""

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from dataclasses_json import DataClassJsonMixin

from llama_index.legacy.data_structs.struct_type import IndexStructType
from llama_index.legacy.schema import BaseNode, TextNode

# TODO: legacy backport of old Node class
Node = TextNode


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a LlamaIndex."""

    index_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: Optional[str] = None

    def get_summary(self) -> str:
        """Get text summary."""
        if self.summary is None:
            raise ValueError("summary field of the index_struct not set.")
        return self.summary

    @classmethod
    @abstractmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""


@dataclass
class IndexGraph(IndexStruct):
    """A graph representing the tree-structured index."""

    # mapping from index in tree to Node doc id.
    all_nodes: Dict[int, str] = field(default_factory=dict)
    root_nodes: Dict[int, str] = field(default_factory=dict)
    node_id_to_children_ids: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def node_id_to_index(self) -> Dict[str, int]:
        """Map from node id to index."""
        return {node_id: index for index, node_id in self.all_nodes.items()}

    @property
    def size(self) -> int:
        """Get the size of the graph."""
        return len(self.all_nodes)

    def get_index(self, node: BaseNode) -> int:
        """Get index of node."""
        return self.node_id_to_index[node.node_id]

    def insert(
        self,
        node: BaseNode,
        index: Optional[int] = None,
        children_nodes: Optional[Sequence[BaseNode]] = None,
    ) -> None:
        """Insert node."""
        index = index or self.size
        node_id = node.node_id

        self.all_nodes[index] = node_id

        if children_nodes is None:
            children_nodes = []
        children_ids = [n.node_id for n in children_nodes]
        self.node_id_to_children_ids[node_id] = children_ids

    def get_children(self, parent_node: Optional[BaseNode]) -> Dict[int, str]:
        """Get children nodes."""
        if parent_node is None:
            return self.root_nodes
        else:
            parent_id = parent_node.node_id
            children_ids = self.node_id_to_children_ids[parent_id]
            return {
                self.node_id_to_index[child_id]: child_id for child_id in children_ids
            }

    def insert_under_parent(
        self,
        node: BaseNode,
        parent_node: Optional[BaseNode],
        new_index: Optional[int] = None,
    ) -> None:
        """Insert under parent node."""
        new_index = new_index or self.size
        if parent_node is None:
            self.root_nodes[new_index] = node.node_id
            self.node_id_to_children_ids[node.node_id] = []
        else:
            if parent_node.node_id not in self.node_id_to_children_ids:
                self.node_id_to_children_ids[parent_node.node_id] = []
            self.node_id_to_children_ids[parent_node.node_id].append(node.node_id)

        self.all_nodes[new_index] = node.node_id

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.TREE


@dataclass
class KeywordTable(IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    table: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add text to table."""
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(node.node_id)

    @property
    def node_ids(self) -> Set[str]:
        """Get all node ids."""
        return set.union(*self.table.values())

    @property
    def keywords(self) -> Set[str]:
        """Get all keywords in the table."""
        return set(self.table.keys())

    @property
    def size(self) -> int:
        """Get the size of the table."""
        return len(self.table)

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.KEYWORD_TABLE


@dataclass
class IndexList(IndexStruct):
    """A list of documents."""

    nodes: List[str] = field(default_factory=list)

    def add_node(self, node: BaseNode) -> None:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        self.nodes.append(node.node_id)

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.LIST


@dataclass
class IndexDict(IndexStruct):
    """A simple dictionary of documents."""

    # TODO: slightly deprecated, should likely be a list or set now
    # mapping from vector store id to node doc_id
    nodes_dict: Dict[str, str] = field(default_factory=dict)

    # TODO: deprecated, not used
    # mapping from node doc_id to vector store id
    doc_id_dict: Dict[str, List[str]] = field(default_factory=dict)

    # TODO: deprecated, not used
    # this should be empty for all other indices
    embeddings_dict: Dict[str, List[float]] = field(default_factory=dict)

    def add_node(
        self,
        node: BaseNode,
        text_id: Optional[str] = None,
    ) -> str:
        """Add text to table, return current position in list."""
        # # don't worry about child indices for now, nodes are all in order
        # self.nodes_dict[int_id] = node
        vector_id = text_id if text_id is not None else node.node_id
        self.nodes_dict[vector_id] = node.node_id

        return vector_id

    def delete(self, doc_id: str) -> None:
        """Delete a Node."""
        del self.nodes_dict[doc_id]

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.VECTOR_STORE


@dataclass
class MultiModelIndexDict(IndexDict):
    """A simple dictionary of documents, but loads a MultiModelVectorStore."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.MULTIMODAL_VECTOR_STORE


@dataclass
class KG(IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    # Unidirectional

    # table of keywords to node ids
    table: Dict[str, Set[str]] = field(default_factory=dict)

    # TODO: legacy attribute, remove in future releases
    rel_map: Dict[str, List[List[str]]] = field(default_factory=dict)

    # TBD, should support vector store, now we just persist the embedding memory
    # maybe chainable abstractions for *_stores could be designed
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def node_ids(self) -> Set[str]:
        """Get all node ids."""
        return set.union(*self.table.values())

    def add_to_embedding_dict(self, triplet_str: str, embedding: List[float]) -> None:
        """Add embedding to dict."""
        self.embedding_dict[triplet_str] = embedding

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add text to table."""
        node_id = node.node_id
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(node_id)

    def search_node_by_keyword(self, keyword: str) -> List[str]:
        """Search for nodes by keyword."""
        if keyword not in self.table:
            return []
        return list(self.table[keyword])

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.KG


@dataclass
class EmptyIndexStruct(IndexStruct):
    """Empty index."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.EMPTY
