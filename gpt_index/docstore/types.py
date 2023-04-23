from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from gpt_index.data_structs.node_v2 import Node

from gpt_index.schema import BaseDocument


class BaseDocumentStore(ABC):
    # ===== Save/load =====
    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        docs_dict: Dict[str, Any],
    ) -> "BaseDocumentStore":
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...

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

    @abstractmethod
    def update_docstore(self, other: "BaseDocumentStore") -> None:
        """Update docstore."""
        ...

    # ===== Hash =====
    @abstractmethod
    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        ...

    @abstractmethod
    def get_document_hash(self, doc_id: str) -> Optional[str]:
        ...

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
