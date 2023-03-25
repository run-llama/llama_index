"""Data structures v2.

Nodes are decoupled from the indices.

"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
from dataclasses_json import DataClassJsonMixin

from pydantic import Json

from gpt_index.data_structs.node_v2 import Node
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.utils import get_new_id



@dataclass
class V2IndexStruct(DataClassJsonMixin):
    """A base data struct for a LlamaIndex."""

    index_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Post init."""
        # assign index_id if not set
        if self.index_id is None:
            self.doc_id = get_new_id(set())

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Document type."""

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        out_dit = {
            'type': self.get_type(),
            'data': super().to_dict(encode_json),
        }
        return out_dit

@dataclass
class IndexGraph(V2IndexStruct):
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

    def get_index(self, node: Node) -> int:
        """Get index of node."""
        return self.node_id_to_index[node.get_doc_id()]

    def insert(
        self,
        node: Node,
        index: Optional[int] = None,
        children_nodes: Optional[Sequence[Node]] = None,
    ) -> None:
        """Insert node."""
        index = index or self.size
        node_id = node.get_doc_id()

        self.all_nodes[index] = node_id

        if children_nodes is None:
            children_nodes = []
        children_ids = [n.get_doc_id() for n in children_nodes]
        self.node_id_to_children_ids[node_id] = children_ids

    def get_children(self, parent_node: Optional[Node]) -> Dict[int, str]:
        """Get children nodes."""
        if parent_node is None:
            return self.root_nodes
        else:
            parent_id = parent_node.get_doc_id()
            children_ids = self.node_id_to_children_ids[parent_id]
            return {
                self.node_id_to_index[child_id]: child_id
                for child_id in children_ids
            }

    def insert_under_parent(
        self, node: Node, parent_node: Optional[Node], new_index: Optional[int] = None
    ) -> None:
        """Insert under parent node."""
        new_index = new_index or self.size
        if parent_node is None:
            self.root_nodes[new_index] = node.get_doc_id()
        else:
            if parent_node.doc_id not in self.node_id_to_children_ids:
                self.node_id_to_children_ids[parent_node.get_doc_id()] = []
            self.node_id_to_children_ids[parent_node.get_doc_id()].append(node.get_doc_id())

        self.all_nodes[new_index] = node.get_doc_id()

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.TREE


@dataclass
class KeywordTable(V2IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    table: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, keywords: List[str], node: Node) -> None:
        """Add text to table."""
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(node.get_doc_id())

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
    def get_type(cls) -> str:
        """Get type."""
        return "keyword_table"


@dataclass
class IndexList(V2IndexStruct):
    """A list of documents."""

    nodes: List[str] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        self.nodes.append(node.get_doc_id())

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.LIST


@dataclass
class IndexDict(V2IndexStruct):
    """A simple dictionary of documents."""

    # nodes_dict: Dict[int, Node] = field(default_factory=dict)
    # id_map: Dict[str, int] = field(default_factory=dict)
    # nodes_set: Set[str] = field(default_factory=set)

    # mapping from vector store id to node id
    nodes_dict: Dict[str, str] = field(default_factory=dict)
    # mapping from doc_id to vector store id
    doc_id_dict: Dict[str, List[str]] = field(default_factory=dict)

    # TODO: temporary hack to store embeddings for simple vector index
    # this should be empty for all other indices
    embeddings_dict: Dict[str, List[float]] = field(default_factory=dict)

    def add_node(
        self,
        node: Node,
        # NOTE: unused
        text_id: Optional[str] = None,
    ) -> str:
        """Add text to table, return current position in list."""
        # int_id = get_new_int_id(set(self.nodes_dict.keys()))
        # if text_id in self.id_map:
        #     raise ValueError("text_id cannot already exist in index.")
        # elif text_id is not None and not isinstance(text_id, str):
        #     raise ValueError("text_id must be a string.")
        # elif text_id is None:
        #     text_id = str(int_id)
        # self.id_map[text_id] = int_id

        # # don't worry about child indices for now, nodes are all in order
        # self.nodes_dict[int_id] = node
        vector_id = text_id if text_id is not None else node.get_doc_id()
        self.nodes_dict[vector_id] = node.get_doc_id()
        if node.ref_doc_id is not None:
            if node.ref_doc_id not in self.doc_id_dict:
                self.doc_id_dict[node.ref_doc_id] = []
            self.doc_id_dict[node.ref_doc_id].append(vector_id)

        return vector_id

    # def get_nodes(self, text_ids: List[str]) -> List[Node]:
    #     """Get nodes."""
    #     nodes = []
    #     for text_id in text_ids:
    #         if text_id not in self.id_map:
    #             raise ValueError("text_id not found in id_map")
    #         elif not isinstance(text_id, str):
    #             raise ValueError("text_id must be a string.")
    #         int_id = self.id_map[text_id]
    #         if int_id not in self.nodes_dict:
    #             raise ValueError("int_id not found in nodes_dict")
    #         nodes.append(self.nodes_dict[int_id])
    #     return nodes

    # def get_node(self, text_id: str) -> Node:
    #     """Get node."""
    #     return self.get_nodes([text_id])[0]

    def delete(self, doc_id: str) -> None:
        """Delete a Node."""
        if doc_id not in self.doc_id_dict:
            raise ValueError("doc_id not found in doc_id_dict")
        for vector_id in self.doc_id_dict[doc_id]:
            del self.nodes_dict[vector_id]

        # for vector_id, node_id in self.nodes_dict.items():
        #     if node_id == doc_id:
        #         del self.nodes_dict[vector_id]
        #         break

        # self.nodes_set.remove(doc_id)
        # if doc_id in self.embeddings_dict:
        #     del self.embeddings_dict[doc_id]

        # text_ids_to_delete = set()
        # int_ids_to_delete = set()
        # for text_id, int_id in self.id_map.items():
        #     node = self.nodes_dict[int_id]
        #     if node.ref_doc_id != doc_id:
        #         continue
        #     text_ids_to_delete.add(text_id)
        #     int_ids_to_delete.add(int_id)

        # for int_id, text_id in zip(int_ids_to_delete, text_ids_to_delete):
        #     del self.nodes_dict[int_id]
        #     del self.id_map[text_id]

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.VECTOR_STORE


@dataclass
class KG(V2IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    # Unidirectional

    table: Dict[str, Set[str]] = field(default_factory=dict)
    # text_chunks: Dict[str, Node] = field(default_factory=dict)
    rel_map: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def node_ids(self) -> Set[str]:
        """Get all node ids."""
        return set.union(*self.table.values())

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
        # self.text_chunks[node_id] = node

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
        return IndexStructType.KG


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


class EmptyIndex(IndexDict):
    """Empty index."""

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.EMPTY

class CompositeIndex(V2IndexStruct):
    all_index_structs: Dict[str, V2IndexStruct]
    root_id: str

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return IndexStructType.COMPOSITE

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        out_dict = {
            "all_index_structs": {
                id_: struct.to_dict(encode_json=encode_json)
                for id_, struct in self.all_index_structs.items()
            }
            "root_id", self.root_id,
        }
        return out_dict