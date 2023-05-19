"""Base index classes."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.data_structs.data_structs import IndexStruct
from llama_index.data_structs.node import Node
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.storage.storage_context import StorageContext
from llama_index.token_counter.token_counter import llm_token_counter

IS = TypeVar("IS", bound=IndexStruct)
IndexType = TypeVar("IndexType", bound="BaseGPTIndex")

logger = logging.getLogger(__name__)


class BaseGPTIndex(Generic[IS], ABC):
    """Base LlamaIndex.

    Args:
        nodes (List[Node]): List of nodes to index
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).

    """

    index_struct_cls: Type[IS]

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IS] = None,
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and nodes is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], Node):
            if isinstance(nodes[0], Document):
                raise ValueError(
                    "The constructor now takes in a list of Node objects. "
                    "Since you are passing in a list of Document objects, "
                    "please use `from_documents` instead."
                )
            else:
                raise ValueError("nodes must be a list of Node objects.")

        self._service_context = service_context or ServiceContext.from_defaults()
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._docstore = self._storage_context.docstore
        self._vector_store = self._storage_context.vector_store

        if index_struct is None:
            assert nodes is not None
            index_struct = self.build_index_from_nodes(nodes)
        self._index_struct = index_struct
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Create index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext.from_defaults()
        docstore = storage_context.docstore

        for doc in documents:
            docstore.set_document_hash(doc.get_doc_id(), doc.get_doc_hash())

        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        return cls(
            nodes=nodes,
            storage_context=storage_context,
            service_context=service_context,
            **kwargs,
        )

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @property
    def index_id(self) -> str:
        """Get the index struct."""
        return self._index_struct.index_id

    def set_index_id(self, index_id: str) -> None:
        """Set the index id.

        NOTE: if you decide to set the index_id on the index_struct manually,
        you will need to explicitly call `add_index_struct` on the `index_store`
        to update the index store.

        .. code-block:: python
            index.index_struct.index_id = index_id
            index.storage_context.index_store.add_index_struct(index.index_struct)

        Args:
            index_id (str): Index id to set.

        """
        # delete the old index struct
        old_id = self._index_struct.index_id
        self._storage_context.index_store.delete_index_struct(old_id)
        # add the new index struct
        self._index_struct.index_id = index_id
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @property
    def docstore(self) -> BaseDocumentStore:
        """Get the docstore corresponding to the index."""
        return self._docstore

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    @property
    def storage_context(self) -> StorageContext:
        return self._storage_context

    @property
    def summary(self) -> str:
        return str(self._index_struct.summary)

    @summary.setter
    def summary(self, new_summary: str) -> None:
        self._index_struct.summary = new_summary
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @abstractmethod
    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> IS:
        """Build the index from nodes."""

    @llm_token_counter("build_index_from_nodes")
    def build_index_from_nodes(self, nodes: Sequence[Node]) -> IS:
        """Build the index from nodes."""
        self._docstore.add_documents(nodes, allow_update=True)
        return self._build_index_from_nodes(nodes)

    @abstractmethod
    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""

    @llm_token_counter("insert")
    def insert_nodes(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        self.docstore.add_documents(nodes, allow_update=True)
        self._insert(nodes, **insert_kwargs)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document."""
        nodes = self.service_context.node_parser.get_nodes_from_documents([document])
        self.insert_nodes(nodes, **insert_kwargs)
        self.docstore.set_document_hash(document.get_doc_id(), document.get_doc_hash())

    @abstractmethod
    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index.

        All nodes in the index related to the index will be deleted.

        Args:
            doc_id (str): document id

        """
        logger.debug(f"> Deleting document: {doc_id}")
        self._delete(doc_id, **delete_kwargs)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def update(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        self.delete(document.get_doc_id(), **update_kwargs.pop("delete_kwargs", {}))
        self.insert(document, **update_kwargs.pop("insert_kwargs", {}))

    def refresh(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or extra_info. It
        will also insert any documents that previously were not stored.
        """
        refreshed_documents = [False] * len(documents)
        for i, document in enumerate(documents):
            existing_doc_hash = self._docstore.get_document_hash(document.get_doc_id())
            if existing_doc_hash != document.get_doc_hash():
                self.update(document, **update_kwargs)
                refreshed_documents[i] = True
            elif existing_doc_hash is None:
                self.insert(document, **update_kwargs.pop("insert_kwargs", {}))
                refreshed_documents[i] = True

        return refreshed_documents

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        pass

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

        retriever = self.as_retriever(**kwargs)

        kwargs["retriever"] = retriever
        if "service_context" not in kwargs:
            kwargs["service_context"] = self._service_context
        return RetrieverQueryEngine.from_args(**kwargs)
