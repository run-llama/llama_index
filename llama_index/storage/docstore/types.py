import os
import fsspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Any, Dict, List, Optional, Sequence
from llama_index.data_structs.node import Node

from llama_index.schema import BaseDocument


DEFAULT_PERSIST_FNAME = "docstore.json"
DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


@dataclass
class RefDocInfo(DataClassJsonMixin):
    """Dataclass to represent ingested documents."""

    doc_ids: List = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)


class BaseDocumentStore(ABC):
    # ===== Save/load =====
    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        pass

    # ===== Main interface =====
    @property
    @abstractmethod
    def docs(self) -> Dict[str, BaseDocument]:
        ...

    @abstractmethod
    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> None:
        ...

    @abstractmethod
    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        ...

    @abstractmethod
    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store."""
        ...

    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        ...

    # ===== Hash =====
    @abstractmethod
    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        ...

    @abstractmethod
    def get_document_hash(self, doc_id: str) -> Optional[str]:
        ...

    # ==== Ref Docs =====
    @abstractmethod
    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents."""

    @abstractmethod
    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""

    @abstractmethod
    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Delete a ref_doc and all it's associated nodes."""

    # ===== Nodes =====
    def get_nodes(self, node_ids: List[str], raise_error: bool = True) -> List[Node]:
        """Get nodes from docstore.

        Args:
            node_ids (List[str]): node ids
            raise_error (bool): raise error if node_id not found

        """
        return [self.get_node(node_id, raise_error=raise_error) for node_id in node_ids]

    def get_node(self, node_id: str, raise_error: bool = True) -> Node:
        """Get node from docstore.

        Args:
            node_id (str): node id
            raise_error (bool): raise error if node_id not found

        """
        doc = self.get_document(node_id, raise_error=raise_error)
        if not isinstance(doc, Node):
            raise ValueError(f"Document {node_id} is not a Node.")
        return doc

    def get_node_dict(self, node_id_dict: Dict[int, str]) -> Dict[int, Node]:
        """Get node dict from docstore given a mapping of index to node ids.

        Args:
            node_id_dict (Dict[int, str]): mapping of index to node ids

        """
        return {
            index: self.get_node(node_id) for index, node_id in node_id_dict.items()
        }
