"""Base vector store index.

An index that that is built on top of an existing vector store.

"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from llama_index.async_utils import run_async_tasks
from llama_index.data_structs.data_structs import IndexDict, IndexStruct
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import async_embed_nodes, embed_nodes
from llama_index.schema import BaseNode, Document, ImageDocument, ImageNode, IndexNode
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import VectorStore

IS = TypeVar("IS", bound=IndexStruct)
IndexType = TypeVar("IndexType", bound="BaseIndex")

logger = logging.getLogger(__name__)


class MultiModalVectorStoreIndex(BaseIndex[IndexDict]):
    """Multi-Modal Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        image_nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDict] = None,
        image_index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._use_async = use_async
        self._store_nodes_override = store_nodes_override
        # super().__init__(
        #     nodes=nodes,
        #     index_struct=index_struct,
        #     service_context=service_context,
        #     storage_context=storage_context,
        #     show_progress=show_progress,
        #     **kwargs,
        # )
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("One of nodes or index_struct must be provided.")
        if index_struct is not None and nodes is not None:
            raise ValueError("Only one of nodes or index_struct can be provided.")
        if image_index_struct is None and image_nodes is None:
            raise ValueError(
                "One of image_nodes or image_index_struct must be provided."
            )
        if image_index_struct is not None and image_nodes is not None:
            raise ValueError(
                "Only one of image_nodes or image_index_struct can be provided."
            )
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], BaseNode):
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
        self._show_progress = show_progress
        self._vector_store = self._storage_context.vector_store
        self._graph_store = self._storage_context.graph_store

        with self._service_context.callback_manager.as_trace("index_construction"):
            if index_struct is None:
                assert nodes is not None
                index_struct = self.build_index_from_nodes(nodes)
            if image_index_struct is None:
                assert image_nodes is not None
                image_index_struct = self.build_index_from_nodes(image_nodes)

            self._index_struct = index_struct
            self._image_index_struct = image_index_struct
            self._storage_context.index_store.add_index_struct(self._index_struct)
            self._storage_context.index_store.add_index_struct(self._image_index_struct)

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Create multi-modal index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext.from_defaults()
        docstore = storage_context.docstore

        with service_context.callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)
            nodes = service_context.node_parser.get_nodes_from_documents(
                documents, show_progress=show_progress
            )
            image_nodes = []
            for doc in documents:
                if isinstance(doc, ImageDocument):
                    image_nodes.append(doc)
            print(len(image_nodes))
            return cls(
                nodes=nodes,
                image_nodes=image_nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=show_progress,
                **kwargs,
            )

    # @classmethod
    # def from_vector_store(
    #     cls,
    #     vector_store: VectorStore,
    #     service_context: Optional[ServiceContext] = None,
    #     **kwargs: Any,
    # ) -> "VectorStoreIndex":
    #     if not vector_store.stores_text:
    #         raise ValueError(
    #             "Cannot initialize from a vector store that does not store text."
    #         )

    #     storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #     return cls(
    #         nodes=[], service_context=service_context, storage_context=storage_context
    #     )

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def image_index_struct(self) -> IndexDict:
        """Get the image index struct."""
        return self._image_index_struct

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.multi_modal.retrievers import (
            MultiModalVectorIndexRetriever,
        )

        return MultiModalVectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            image_node_ids=list(self.image_index_struct.nodes_dict.values()),
            **kwargs,
        )

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = embed_nodes(
            nodes, self._service_context.embed_model, show_progress=show_progress
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            results.append(result)
        return results

    def _get_node_with_image_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, image_node, and image embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = embed_nodes(
            nodes, self._service_context.image_embed_model, show_progress=show_progress
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = await async_embed_nodes(
            nodes=nodes,
            embed_model=self._service_context.embed_model,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            results.append(result)
        return results

    async def _async_add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        nodes = await self._aget_node_with_embedding(nodes, show_progress)
        new_ids = await self._vector_store.async_add(nodes)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(nodes, new_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for node, new_id in zip(nodes, new_ids):
                if isinstance(node, (ImageNode, IndexNode)):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        nodes = self._get_node_with_embedding(nodes, show_progress)
        new_ids = self._vector_store.add(nodes)

        if not self._vector_store.stores_text or self._store_nodes_override:
            # NOTE: if the vector store doesn't store text,
            # we need to add the nodes to the index struct and document store
            for node, new_id in zip(nodes, new_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for node, new_id in zip(nodes, new_ids):
                if isinstance(node, (ImageNode, IndexNode)):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    def _add_image_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add image document to index."""
        if not nodes:
            return

        nodes = self._get_node_with_embedding(nodes, show_progress)
        new_ids = self._image_store.add(nodes)

        if not self._image_store.stores_text or self._store_nodes_override:
            # NOTE: if the vector store doesn't store text,
            # we need to add the nodes to the index struct and document store
            for node, new_id in zip(nodes, new_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for node, new_id in zip(nodes, new_ids):
                if isinstance(node, (ImageNode, IndexNode)):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [
                self._async_add_nodes_to_index(
                    index_struct, nodes, show_progress=self._show_progress
                )
            ]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(
                index_struct, nodes, show_progress=self._show_progress
            )
        return index_struct

    def _build_index_from_image_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [
                self._async_add_image_nodes_to_index(
                    index_struct, nodes, show_progress=self._show_progress
                )
            ]
            run_async_tasks(tasks)
        else:
            self._add_image_nodes_to_index(
                index_struct, nodes, show_progress=self._show_progress
            )
        return index_struct

    def build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build the index from nodes.

        NOTE: Overrides BaseIndex.build_index_from_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        return self._build_index_from_nodes(nodes)

    def build_index_from_image_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build the index from nodes.

        NOTE: Overrides BaseIndex.build_index_from_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        return self._build_index_from_image_nodes(nodes)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes)

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes.

        NOTE: overrides BaseIndex.insert_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        self._insert(nodes, **insert_kwargs)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        pass

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a list of nodes from the index.

        Args:
            doc_ids (List[str]): A list of doc_ids from the nodes to delete

        """
        raise NotImplementedError(
            "Vector indices currently only support delete_ref_doc, which "
            "deletes nodes using the ref_doc_id of ingested documents."
        )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        self._vector_store.delete(ref_doc_id)

        # delete from index_struct only if needed
        if not self._vector_store.stores_text or self._store_nodes_override:
            ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
            if ref_doc_info is not None:
                for node_id in ref_doc_info.node_ids:
                    self._index_struct.delete(node_id)
                    self._vector_store.delete(node_id)

        # delete from docstore only if needed
        if (
            not self._vector_store.stores_text or self._store_nodes_override
        ) and delete_from_docstore:
            self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        if not self._vector_store.stores_text or self._store_nodes_override:
            node_doc_ids = list(self.index_struct.nodes_dict.values())
            nodes = self.docstore.get_nodes(node_doc_ids)

            all_ref_doc_info = {}
            for node in nodes:
                ref_node = node.source_node
                if not ref_node:
                    continue

                ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
                if not ref_doc_info:
                    continue

                all_ref_doc_info[ref_node.node_id] = ref_doc_info
            return all_ref_doc_info
        else:
            raise NotImplementedError(
                "Vector store integrations that store text in the vector store are "
                "not supported by ref_doc_info yet."
            )
