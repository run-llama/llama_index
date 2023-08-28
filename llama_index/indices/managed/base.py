"""Base Managed Service index.

An index that that is built on top of a managed service.

"""
from abc import ABC, abstractmethod

from typing import Any, Dict, List, Optional, Sequence, Type

from llama_index.data_structs.data_structs import IndexDict, IndexStruct
from llama_index.indices.base import BaseIndex, IS, IndexType
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.schema import BaseNode, Document
from llama_index.storage.docstore.types import RefDocInfo


class ManagedIndex(BaseIndex[IndexDict], ABC):
    """Managed Index.
    The managed service can index documents into a managed service.
    How documents are structured into nodes is a detail for the managed service, and not exposed in this interface (although could be controlled by configuration parameters)

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexStruct] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=None,
            storage_context=None,
            show_progress=show_progress,
            **kwargs,
        )

    @abstractmethod
    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document."""

    @abstractmethod
    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""

    @abstractmethod
    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes."""

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build the index from nodes."""
        raise NotImplementedError(
            "_build_index_from_nodes not supported for a Managed index."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("_delete_node not supported for a Managed index.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError("ref_doc_info not supported for a Managed index.")

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Build an index from a sequence of documents."""
