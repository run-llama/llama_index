"""Multi Modal Vector Store Index.

An index that is built on top of multiple vector stores for different modalities.

"""

import logging
from typing import Any, List, Optional, Sequence, cast

from llama_index.legacy.core.base_query_engine import BaseQueryEngine
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.data_structs.data_structs import IndexDict, MultiModelIndexDict
from llama_index.legacy.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.legacy.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.legacy.indices.utils import (
    async_embed_image_nodes,
    async_embed_nodes,
    embed_image_nodes,
    embed_nodes,
)
from llama_index.legacy.indices.vector_store.base import VectorStoreIndex
from llama_index.legacy.schema import BaseNode, ImageNode
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.storage_context import StorageContext
from llama_index.legacy.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
    SimpleVectorStore,
)
from llama_index.legacy.vector_stores.types import VectorStore

logger = logging.getLogger(__name__)


class MultiModalVectorStoreIndex(VectorStoreIndex):
    """Multi-Modal Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    image_namespace = "image"
    index_struct_cls = MultiModelIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[MultiModelIndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        # Image-related kwargs
        # image_vector_store going to be deprecated. image_store can be passed from storage_context
        # keep image_vector_store here for backward compatibility
        image_vector_store: Optional[VectorStore] = None,
        image_embed_model: EmbedType = "clip",
        is_image_to_text: bool = False,
        # is_image_vector_store_empty is used to indicate whether image_vector_store is empty
        # those flags are used for cases when only one vector store is used
        is_image_vector_store_empty: bool = False,
        is_text_vector_store_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        image_embed_model = resolve_embed_model(image_embed_model)
        assert isinstance(image_embed_model, MultiModalEmbedding)
        self._image_embed_model = image_embed_model
        self._is_image_to_text = is_image_to_text
        self._is_image_vector_store_empty = is_image_vector_store_empty
        self._is_text_vector_store_empty = is_text_vector_store_empty
        storage_context = storage_context or StorageContext.from_defaults()

        if image_vector_store is not None:
            if self.image_namespace not in storage_context.vector_stores:
                storage_context.add_vector_store(
                    image_vector_store, self.image_namespace
                )
            else:
                # overwrite image_store from storage_context
                storage_context.vector_stores[self.image_namespace] = image_vector_store

        if self.image_namespace not in storage_context.vector_stores:
            storage_context.add_vector_store(SimpleVectorStore(), self.image_namespace)

        self._image_vector_store = storage_context.vector_stores[self.image_namespace]

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            **kwargs,
        )

    @property
    def image_vector_store(self) -> VectorStore:
        return self._image_vector_store

    @property
    def image_embed_model(self) -> MultiModalEmbedding:
        return self._image_embed_model

    @property
    def is_image_vector_store_empty(self) -> bool:
        return self._is_image_vector_store_empty

    @property
    def is_text_vector_store_empty(self) -> bool:
        return self._is_text_vector_store_empty

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.legacy.indices.multi_modal.retriever import (
            MultiModalVectorIndexRetriever,
        )

        return MultiModalVectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            **kwargs,
        )

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        """As query engine."""
        from llama_index.legacy.indices.multi_modal.retriever import (
            MultiModalVectorIndexRetriever,
        )
        from llama_index.legacy.query_engine.multi_modal import (
            SimpleMultiModalQueryEngine,
        )

        retriever = cast(MultiModalVectorIndexRetriever, self.as_retriever(**kwargs))

        return SimpleMultiModalQueryEngine(
            retriever,
            **kwargs,
        )

    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore,
        service_context: Optional[ServiceContext] = None,
        # Image-related kwargs
        image_vector_store: Optional[VectorStore] = None,
        image_embed_model: EmbedType = "clip",
        **kwargs: Any,
    ) -> "VectorStoreIndex":
        if not vector_store.stores_text:
            raise ValueError(
                "Cannot initialize from a vector store that does not store text."
            )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return cls(
            nodes=[],
            service_context=service_context,
            storage_context=storage_context,
            image_vector_store=image_vector_store,
            image_embed_model=image_embed_model,
            **kwargs,
        )

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        is_image: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_text_embed_map = None

        if is_image:
            id_to_embed_map = embed_image_nodes(
                nodes,
                embed_model=self._image_embed_model,
                show_progress=show_progress,
            )

            # text field is populate, so embed them
            if self._is_image_to_text:
                id_to_text_embed_map = embed_nodes(
                    nodes,
                    embed_model=self._service_context.embed_model,
                    show_progress=show_progress,
                )
                # TODO: refactor this change of image embed model to same as text
                self._image_embed_model = self._service_context.embed_model

        else:
            id_to_embed_map = embed_nodes(
                nodes,
                embed_model=self._service_context.embed_model,
                show_progress=show_progress,
            )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            if is_image and id_to_text_embed_map:
                text_embedding = id_to_text_embed_map[node.node_id]
                result.text_embedding = text_embedding
                result.embedding = (
                    text_embedding  # TODO: re-factor to make use of both embeddings
                )
            results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        is_image: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_text_embed_map = None

        if is_image:
            id_to_embed_map = await async_embed_image_nodes(
                nodes,
                embed_model=self._image_embed_model,
                show_progress=show_progress,
            )

            if self._is_image_to_text:
                id_to_text_embed_map = await async_embed_nodes(
                    nodes,
                    embed_model=self._service_context.embed_model,
                    show_progress=show_progress,
                )
                # TODO: refactor this change of image embed model to same as text
                self._image_embed_model = self._service_context.embed_model

        else:
            id_to_embed_map = await async_embed_nodes(
                nodes,
                embed_model=self._service_context.embed_model,
                show_progress=show_progress,
            )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            if is_image and id_to_text_embed_map:
                text_embedding = id_to_text_embed_map[node.node_id]
                result.text_embedding = text_embedding
                result.embedding = (
                    text_embedding  # TODO: re-factor to make use of both embeddings
                )
            results.append(result)
        return results

    async def _async_add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        image_nodes: List[ImageNode] = []
        text_nodes: List[BaseNode] = []
        new_text_ids: List[str] = []
        new_img_ids: List[str] = []

        for node in nodes:
            if isinstance(node, ImageNode):
                image_nodes.append(node)
            if node.text:
                text_nodes.append(node)

        if len(text_nodes) > 0:
            # embed all nodes as text - include image nodes that have text attached
            text_nodes = await self._aget_node_with_embedding(
                text_nodes, show_progress, is_image=False
            )
            new_text_ids = await self.storage_context.vector_stores[
                DEFAULT_VECTOR_STORE
            ].async_add(text_nodes, **insert_kwargs)
        else:
            self._is_text_vector_store_empty = True

        if len(image_nodes) > 0:
            # embed image nodes as images directly
            image_nodes = await self._aget_node_with_embedding(
                image_nodes,
                show_progress,
                is_image=True,
            )
            new_img_ids = await self.storage_context.vector_stores[
                self.image_namespace
            ].async_add(image_nodes, **insert_kwargs)
        else:
            self._is_image_vector_store_empty = True

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        all_nodes = text_nodes + image_nodes
        all_new_ids = new_text_ids + new_img_ids
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(all_nodes, all_new_ids):
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
        **insert_kwargs: Any,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        image_nodes: List[ImageNode] = []
        text_nodes: List[BaseNode] = []
        new_text_ids: List[str] = []
        new_img_ids: List[str] = []

        for node in nodes:
            if isinstance(node, ImageNode):
                image_nodes.append(node)
            if node.text:
                text_nodes.append(node)

        if len(text_nodes) > 0:
            # embed all nodes as text - include image nodes that have text attached
            text_nodes = self._get_node_with_embedding(
                text_nodes, show_progress, is_image=False
            )
            new_text_ids = self.storage_context.vector_stores[DEFAULT_VECTOR_STORE].add(
                text_nodes, **insert_kwargs
            )
        else:
            self._is_text_vector_store_empty = True

        if len(image_nodes) > 0:
            # embed image nodes as images directly
            # check if we should use text embedding for images instead of default
            image_nodes = self._get_node_with_embedding(
                image_nodes,
                show_progress,
                is_image=True,
            )
            new_img_ids = self.storage_context.vector_stores[self.image_namespace].add(
                image_nodes, **insert_kwargs
            )
        else:
            self._is_image_vector_store_empty = True

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        all_nodes = text_nodes + image_nodes
        all_new_ids = new_text_ids + new_img_ids
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(all_nodes, all_new_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        # delete from all vector stores

        for vector_store in self._storage_context.vector_stores.values():
            vector_store.delete(ref_doc_id)

            if self._store_nodes_override or self._vector_store.stores_text:
                ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
                if ref_doc_info is not None:
                    for node_id in ref_doc_info.node_ids:
                        self._index_struct.delete(node_id)
                        self._vector_store.delete(node_id)

        if delete_from_docstore:
            self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)
