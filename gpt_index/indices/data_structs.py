"""File for core data structures."""

import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dataclasses_json import DataClassJsonMixin

from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_int_id


@dataclass
class IndexStruct(BaseDocument, DataClassJsonMixin):
    """A base data struct for a GPT index."""

    # NOTE: the text field, inherited from BaseDocument,
    # represents a summary of the content of the index struct.
    # primarily used for composing indices with other indices

    # NOTE: the doc_id field, inherited from BaseDocument,
    # represents a unique identifier for the index struct
    # that will be put in the docstore.
    # Not all index_structs need to have a doc_id. Only index_structs that
    # represent a complete data structure (e.g. IndexGraph, IndexList),
    # and are used to compose a higher level index, will have a doc_id.


@dataclass
class Node(IndexStruct):
    """A generic node of data.

    Used in the GPT Tree Index and List Index.

    """

    def __post_init__(self) -> None:
        """Post init."""
        # NOTE: for Node objects, the text field is required
        if self.text is None:
            raise ValueError("text field not set.")

    # used for GPTTreeIndex
    index: int = 0
    child_indices: Set[int] = field(default_factory=set)

    # embeddings
    embedding: Optional[List[float]] = None

    # reference document id
    ref_doc_id: Optional[str] = None


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
    text_chunks: Dict[int, Node] = field(default_factory=dict)

    def _get_index(self) -> int:
        """Get the next index for the text chunk."""
        # randomly generate until we get a unique index
        while True:
            idx = random.randint(0, sys.maxsize)
            if idx not in self.text_chunks:
                break
        return idx

    def add_text(self, keywords: List[str], text_chunk: str, ref_doc_id: str) -> int:
        """Add text to table."""
        cur_idx = self._get_index()
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(cur_idx)
        self.text_chunks[cur_idx] = Node(text_chunk, ref_doc_id=ref_doc_id)
        return cur_idx

    def get_texts(self, keyword: str) -> List[str]:
        """Get texts given keyword."""
        if keyword not in self.table:
            raise ValueError("Keyword not found in table.")
        return [self.text_chunks[idx].get_text() for idx in self.table[keyword]]

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

    def add_text(self, text_chunk: str, ref_doc_id: str) -> int:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        cur_node = Node(text_chunk, index=len(self.nodes), ref_doc_id=ref_doc_id)
        self.nodes.append(cur_node)
        return cur_node.index


# TODO: this should be specific to FAISS
@dataclass
class IndexDict(IndexStruct):
    """A simple dictionary of documents."""

    nodes_dict: Dict[int, Node] = field(default_factory=dict)
    # TODO: allow users to pass in Hashable type
    # currently Hashable doesn't play nice with dataclasses_json
    id_map: Dict[Any, int] = field(default_factory=dict)

    def add_text(
        self,
        text_chunk: str,
        ref_doc_id: str,
        text_id: Optional[Any] = None,
    ) -> Any:
        """Add text to table, return current position in list."""
        if text_id in self.id_map:
            raise ValueError("text_id cannot already exist in index.")
        int_id = get_new_int_id(set(self.nodes_dict.keys()))
        if text_id is None:
            text_id = int_id
        self.id_map[text_id] = int_id

        # don't worry about child indices for now, nodes are all in order
        cur_node = Node(text_chunk, index=int_id, ref_doc_id=ref_doc_id)
        self.nodes_dict[int_id] = cur_node
        return text_id

    def get_nodes(self, text_ids: List[Any]) -> List[Node]:
        """Get nodes."""
        nodes = []
        for text_id in text_ids:
            if text_id not in self.id_map:
                raise ValueError("text_id not found in id_map")
            int_id = self.id_map[text_id]
            if int_id not in self.nodes_dict:
                raise ValueError("int_id not found in nodes_dict")
            nodes.append(self.nodes_dict[int_id])
        return nodes

    def get_node(self, text_id: Any) -> Node:
        """Get node."""
        return self.get_nodes([text_id])[0]


class IndexStructType(str, Enum):
    """Index struct type."""

    TREE = "tree"
    LIST = "list"
    KEYWORD_TABLE = "keyword_table"
    DICT = "dict"

    def get_index_struct_cls(self) -> type:
        """Get index struct class."""
        if self == IndexStructType.TREE:
            return IndexGraph
        elif self == IndexStructType.LIST:
            return IndexList
        elif self == IndexStructType.KEYWORD_TABLE:
            return KeywordTable
        elif self == IndexStructType.DICT:
            return IndexDict
        else:
            raise ValueError("Invalid index struct type.")

    @classmethod
    def from_index_struct(cls, index_struct: IndexStruct) -> "IndexStructType":
        """Get index enum from index struct class."""
        if isinstance(index_struct, IndexGraph):
            return cls.TREE
        elif isinstance(index_struct, IndexList):
            return cls.LIST
        elif isinstance(index_struct, KeywordTable):
            return cls.KEYWORD_TABLE
        elif isinstance(index_struct, IndexDict):
            return cls.DICT
        else:
            raise ValueError("Invalid index struct type.")
