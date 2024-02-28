"""Base Managed Service index.

An index that is built on top of a managed service.

"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex, IndexType
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext


class BaseManagedIndex(BaseIndex[IndexDict], ABC):
    """Managed Index.
    The managed service can index documents into a managed service.
    How documents are structured into nodes is a detail for the managed service,
    and not exposed in this interface (although could be controlled by
    configuration parameters).

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        # deprecated
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    @abstractmethod
    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of documents (each a node)."""

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
            "_build_index_from_nodes not implemented for BaseManagedIndex."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("_delete_node not implemented for BaseManagedIndex.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError("ref_doc_info not implemented for BaseManagedIndex.")

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        # deprecated
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build an index from a sequence of documents."""
        raise NotImplementedError(
            "from_documents not implemented for BaseManagedIndex."
        )
