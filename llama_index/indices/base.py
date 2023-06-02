"""Base index classes."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.data_structs.data_structs import IndexStruct
from llama_index.data_structs.node import Node
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from llama_index.storage.docstore.types import BaseDocumentStore, RefDocInfo
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

        with self._service_context.callback_manager.as_trace("index_construction"):
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

        with service_context.callback_manager.as_trace("index_construction"):
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
        with self._service_context.callback_manager.as_trace("insert_nodes"):
            self.docstore.add_documents(nodes, allow_update=True)
            self._insert(nodes, **insert_kwargs)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document."""
        with self._service_context.callback_manager.as_trace("insert"):
            nodes = self.service_context.node_parser.get_nodes_from_documents(
                [document]
            )
            self.insert_nodes(nodes, **insert_kwargs)
            self.docstore.set_document_hash(
                document.get_doc_id(), document.get_doc_hash()
            )

    @abstractmethod
    def _delete_node(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""

    def delete_nodes(
        self,
        doc_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a list of nodes from the index.

        Args:
            doc_ids (List[str]): A list of doc_ids from the nodes to delete

        """
        for doc_id in doc_ids:
            self._delete_node(doc_id, **delete_kwargs)
            if delete_from_docstore:
                self.docstore.delete_document(doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index.
        All nodes in the index related to the index will be deleted.

        Args:
            doc_id (str): A doc_id of the ingested document

        """
        logger.warning(
            "delete() is now deprecated, please refer to delete_ref_doc() to delete "
            "ingested documents+nodes or delete_nodes to delete a list of nodes."
        )
        self.delete_ref_doc(doc_id)

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
        if ref_doc_info is None:
            logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            return

        self.delete_nodes(
            ref_doc_info.doc_ids,
            delete_from_docstore=delete_from_docstore,
            **delete_kwargs,
        )

        if delete_from_docstore:
            self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

    def update(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        logger.warning(
            "update() is now deprecated, please refer to update_ref_doc() to update "
            "ingested documents+nodes."
        )
        self.update_ref_doc(document, **update_kwargs)

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        with self._service_context.callback_manager.as_trace("update"):
            self.delete_ref_doc(
                document.get_doc_id(), **update_kwargs.pop("delete_kwargs", {})
            )
            self.insert(document, **update_kwargs.pop("insert_kwargs", {}))

    def refresh(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or extra_info. It
        will also insert any documents that previously were not stored.
        """
        logger.warning(
            "refresh() is now deprecated, please refer to refresh_ref_docs() to "
            "refresh ingested documents+nodes with an updated list of documents."
        )
        return self.refresh_ref_docs(documents, **update_kwargs)

    def refresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or extra_info. It
        will also insert any documents that previously were not stored.
        """
        with self._service_context.callback_manager.as_trace("refresh"):
            refreshed_documents = [False] * len(documents)
            for i, document in enumerate(documents):
                existing_doc_hash = self._docstore.get_document_hash(
                    document.get_doc_id()
                )
                if existing_doc_hash != document.get_doc_hash():
                    self.update_ref_doc(
                        document, **update_kwargs.pop("update_kwargs", {})
                    )
                    refreshed_documents[i] = True
                elif existing_doc_hash is None:
                    self.insert(document, **update_kwargs.pop("insert_kwargs", {}))
                    refreshed_documents[i] = True

            return refreshed_documents

    @property
    @abstractmethod
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        ...

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

    def as_chat_engine(
        self, chat_mode: ChatMode = ChatMode.CONDENSE_QUESTION, **kwargs: Any
    ) -> BaseChatEngine:
        if chat_mode == ChatMode.CONDENSE_QUESTION:
            # NOTE: lazy import
            from llama_index.chat_engine import CondenseQuestionChatEngine

            query_engine = self.as_query_engine(**kwargs)
            return CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine, **kwargs
            )
        elif chat_mode == ChatMode.REACT:
            # NOTE: lazy import
            from llama_index.chat_engine import ReActChatEngine

            query_engine = self.as_query_engine(**kwargs)
            return ReActChatEngine.from_query_engine(
                query_engine=query_engine, **kwargs
            )
        else:
            raise ValueError(f"Unknown chat mode: {chat_mode}")
