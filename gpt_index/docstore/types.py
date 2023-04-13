from abc import ABC
from typing import Any, Dict, List, Optional, Sequence
from gpt_index.data_structs.node_v2 import Node

from gpt_index.schema import BaseDocument


class DocumentStore(ABC):
    # ===== Constructors =====
    @classmethod
    def from_documents(
        cls, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> "DocumentStore":
        ...

    # ===== Main interface =====
    @property
    def docs(self) -> Dict[str, BaseDocument]:
        ...

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> None:
        ...

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        ...

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store."""
        ...

    def document_exists(self, doc_id: str) -> bool:
        ...

    def update_docstore(self, other: "DocumentStore") -> None:
        """Update docstore."""
        ...

    # ===== Hash =====
    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        ...

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        ...

    # ===== Nodes =====
    def get_nodes(self, node_ids: List[str], raise_error: bool = True) -> List[Node]:
        ...

    def get_node(self, node_id: str, raise_error: bool = True) -> Node:
        ...

    def get_node_dict(self, node_id_dict: Dict[int, str]) -> Dict[int, Node]:
        ...

    # ===== Save/load =====
    @classmethod
    def from_dict(
        cls,
        docs_dict: Dict[str, Any],
    ) -> "DocumentStore":
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...
