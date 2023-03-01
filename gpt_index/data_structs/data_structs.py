"""File for core data structures."""

import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_int_id


@dataclass
class IndexStruct(BaseDocument, DataClassJsonMixin):
    """A base data struct for a LlamaIndex."""

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

    Base struct used in most indices.

    """

    def __post_init__(self) -> None:
        """Post init."""
        super().__post_init__()
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

    # extra node info
    node_info: Optional[Dict[str, Any]] = None

    def get_text(self) -> str:
        """Get text."""
        text = super().get_text()
        result_text = (
            text if self.extra_info_str is None else f"{self.extra_info_str}\n\n{text}"
        )
        return result_text

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        # TODO: consolidate with IndexStructType
        return "node"


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

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return "tree"


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

    def add_node(self, keywords: List[str], node: Node) -> int:
        """Add text to table."""
        cur_idx = self._get_index()
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(cur_idx)
        self.text_chunks[cur_idx] = node
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

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return "keyword_table"


@dataclass
class IndexList(IndexStruct):
    """A list of documents."""

    nodes: List[Node] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        self.nodes.append(node)

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return "list"


@dataclass
class IndexDict(IndexStruct):
    """A simple dictionary of documents."""

    nodes_dict: Dict[int, Node] = field(default_factory=dict)
    id_map: Dict[str, int] = field(default_factory=dict)

    # TODO: temporary hack to store embeddings for simple vector index
    # this should be empty for all other indices
    embeddings_dict: Dict[str, List[float]] = field(default_factory=dict)

    def add_node(
        self,
        node: Node,
        text_id: Optional[str] = None,
    ) -> str:
        """Add text to table, return current position in list."""
        int_id = get_new_int_id(set(self.nodes_dict.keys()))
        if text_id in self.id_map:
            raise ValueError("text_id cannot already exist in index.")
        elif text_id is not None and not isinstance(text_id, str):
            raise ValueError("text_id must be a string.")
        elif text_id is None:
            text_id = str(int_id)
        self.id_map[text_id] = int_id

        # don't worry about child indices for now, nodes are all in order
        self.nodes_dict[int_id] = node
        return text_id

    def get_nodes(self, text_ids: List[str]) -> List[Node]:
        """Get nodes."""
        nodes = []
        for text_id in text_ids:
            if text_id not in self.id_map:
                raise ValueError("text_id not found in id_map")
            elif not isinstance(text_id, str):
                raise ValueError("text_id must be a string.")
            int_id = self.id_map[text_id]
            if int_id not in self.nodes_dict:
                raise ValueError("int_id not found in nodes_dict")
            nodes.append(self.nodes_dict[int_id])
        return nodes

    def get_node(self, text_id: str) -> Node:
        """Get node."""
        return self.get_nodes([text_id])[0]

    def delete(self, doc_id: str) -> None:
        """Delete a document."""
        text_ids_to_delete = set()
        int_ids_to_delete = set()
        for text_id, int_id in self.id_map.items():
            node = self.nodes_dict[int_id]
            if node.ref_doc_id != doc_id:
                continue
            text_ids_to_delete.add(text_id)
            int_ids_to_delete.add(int_id)

        for int_id, text_id in zip(int_ids_to_delete, text_ids_to_delete):
            del self.nodes_dict[int_id]
            del self.id_map[text_id]

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.VECTOR_STORE


@dataclass
class KG(IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    # Unidirectional

    table: Dict[str, Set[str]] = field(default_factory=dict)
    text_chunks: Dict[str, Node] = field(default_factory=dict)
    rel_map: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)

    def add_to_embedding_dict(self, triplet_str: str, embedding: List[float]) -> None:
        """Add embedding to dict."""
        self.embedding_dict[triplet_str] = embedding

    def upsert_triplet(self, triplet: Tuple[str, str, str], node: Node) -> None:
        """Upsert a knowledge triplet to the graph."""
        subj, relationship, obj = triplet
        self.add_node([subj, obj], node)
        if subj not in self.rel_map:
            self.rel_map[subj] = []
        self.rel_map[subj].append((obj, relationship))

    def add_node(self, keywords: List[str], node: Node) -> None:
        """Add text to table."""
        node_id = node.get_doc_id()
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(node_id)
        self.text_chunks[node_id] = node

    def get_rel_map_texts(self, keyword: str) -> List[str]:
        """Get the corresponding knowledge for a given keyword."""
        # NOTE: return a single node for now
        if keyword not in self.rel_map:
            return []
        texts = []
        for obj, rel in self.rel_map[keyword]:
            texts.append(str((keyword, rel, obj)))
        return texts

    def get_rel_map_tuples(self, keyword: str) -> List[Tuple[str, str]]:
        """Get the corresponding knowledge for a given keyword."""
        # NOTE: return a single node for now
        if keyword not in self.rel_map:
            return []
        return self.rel_map[keyword]

    def get_node_ids(self, keyword: str, depth: int = 1) -> List[str]:
        """Get the corresponding knowledge for a given keyword."""
        if depth > 1:
            raise ValueError("Depth > 1 not supported yet.")
        if keyword not in self.table:
            return []
        keywords = [keyword]
        # some keywords may correspond to a leaf node, may not be in rel_map
        if keyword in self.rel_map:
            keywords.extend([child for child, _ in self.rel_map[keyword]])

        node_ids: List[str] = []
        for keyword in keywords:
            for node_id in self.table.get(keyword, set()):
                node_ids.append(node_id)
            # TODO: Traverse (with depth > 1)
        return node_ids

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return "kg"


# TODO: remove once we centralize UX around vector index


class SimpleIndexDict(IndexDict):
    """Index dict for simple vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.SIMPLE_DICT


class FaissIndexDict(IndexDict):
    """Index dict for Faiss vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.DICT


class WeaviateIndexDict(IndexDict):
    """Index dict for Weaviate vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.WEAVIATE


class PineconeIndexDict(IndexDict):
    """Index dict for Pinecone vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.PINECONE


class QdrantIndexDict(IndexDict):
    """Index dict for Qdrant vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.QDRANT


class ChromaIndexDict(IndexDict):
    """Index dict for Chroma vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.CHROMA


class OpensearchIndexDict(IndexDict):
    """Index dict for Opensearch vector index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.OPENSEARCH
