import logging
from collections import defaultdict
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.data_structs.data_structs import IndexDict, MultiModelIndexDict
from llama_index.core.embeddings.omni_modal_base import (
    KD,
    KQ,
    Modality,
    Modalities,
    OmniModalEmbeddingBundle,
    OmniModalEmbedding,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    MetadataMode,
    TransformComponent,
)
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

logger = logging.getLogger(__name__)


class OmniModalVectorStoreIndex(BaseIndex[MultiModelIndexDict], Generic[KD, KQ]):
    """Omni-Modal Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    index_struct_cls = MultiModelIndexDict

    @staticmethod
    def get_multi_modal_embed_model(index: MultiModalVectorStoreIndex):
        return OmniModalEmbeddingBundle.of(
            OmniModalEmbedding.from_base(
                index._embed_model,
            ),
            OmniModalEmbedding.from_multi_modal(
                index.image_embed_model,
                is_image_to_text=index._is_image_to_text,
            ),
        )

    @staticmethod
    def from_multi_modal(index: MultiModalVectorStoreIndex):
        assert isinstance(index.index_struct, MultiModelIndexDict)

        # Avoid creating a brand new vector index for text modality
        vector_stores = index.storage_context.vector_stores

        return OmniModalVectorStoreIndex(
            embed_model=OmniModalVectorStoreIndex.get_multi_modal_embed_model(index),
            vector_stores={
                Modalities.TEXT.key: vector_stores["default"],
                Modalities.IMAGE.key: index.image_vector_store,
            },
            index_struct=index.index_struct,
            storage_context=index.storage_context,
            callback_manager=index._callback_manager,
            use_async=index._use_async,
            store_nodes_override=index._store_nodes_override,
            show_progress=index._show_progress,
        )

    @staticmethod
    def load_from_storage(
        storage_context: StorageContext,
        embed_model: OmniModalEmbeddingBundle[KD, KQ],
        vector_stores: Optional[Mapping[KD, BasePydanticVectorStore]] = None,
        # base index params
        objects: Optional[Sequence[IndexNode]] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        # vector store index params
        use_async: bool = False,
        store_nodes_override: bool = False,
        insert_batch_size: int = 2048,
        **kwargs: Any,
    ) -> "OmniModalVectorStoreIndex[KD, KQ]":
        """This should be used instead of `load_index_from_storage`."""
        index_structs = storage_context.index_store.index_structs()
        if len(index_structs) > 1:
            msg = f"Expected to load a single index, but got {len(index_structs)} instead."
            raise ValueError(msg)

        (index_struct,) = index_structs
        assert isinstance(index_struct, MultiModelIndexDict)

        return OmniModalVectorStoreIndex(
            embed_model=embed_model,
            vector_stores=vector_stores,
            objects=objects,
            index_struct=index_struct,
            storage_context=storage_context,
            callback_manager=callback_manager,
            transformations=transformations,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            insert_batch_size=insert_batch_size,
            **kwargs,
        )

    @classmethod
    def from_vector_store(
        cls,
        embed_model: OmniModalEmbeddingBundle[KD, KQ],
        vector_stores: Optional[Mapping[KD, BasePydanticVectorStore]] = None,
        # base index params
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[MultiModelIndexDict] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        # vector store index params
        use_async: bool = False,
        store_nodes_override: bool = False,
        insert_batch_size: int = 2048,
        **kwargs: Any,
    ) -> "OmniModalVectorStoreIndex[KD, KQ]":
        return OmniModalVectorStoreIndex(
            embed_model=embed_model,
            vector_stores=vector_stores,
            nodes=None,
            objects=objects,
            index_struct=index_struct,
            storage_context=None,
            callback_manager=callback_manager,
            transformations=transformations,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            insert_batch_size=insert_batch_size,
            **kwargs,
        )

    _index_struct: MultiModelIndexDict

    def __init__(
        self,
        embed_model: OmniModalEmbeddingBundle[KD, KQ],
        # The vector store to use for each document modality, defaulting to SimpleVectorStore
        vector_stores: Optional[Mapping[KD, BasePydanticVectorStore]] = None,
        # base index params
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[MultiModelIndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        # vector store index params
        use_async: bool = False,
        store_nodes_override: bool = False,
        insert_batch_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if vector_stores is None:
            vector_stores = {}

        storage_context = storage_context or StorageContext.from_defaults()
        callback_manager = (
            callback_manager
            or callback_manager_from_settings_or_context(Settings, None)
        )

        self._use_async = use_async
        self._store_nodes_override = store_nodes_override
        self._insert_batch_size = insert_batch_size

        for modality_key in embed_model.document_modalities:
            if modality_key in storage_context.vector_stores:
                logger.info(
                    "A vector store is already initialized for modality (%s), "
                    "so the existing one will be used.",
                    modality_key,
                )
                continue

            if modality_key in vector_stores:
                vector_store = vector_stores[modality_key]
            else:
                vector_store = SimpleVectorStore()

            storage_context.add_vector_store(vector_store, modality_key)

        self._vector_stores = {
            modality_key: storage_context.vector_stores[modality_key]
            for modality_key in embed_model.document_modalities
        }
        self._embed_model = embed_model

        super().__init__(
            nodes=nodes,
            objects=objects,
            index_struct=index_struct,
            storage_context=storage_context,
            callback_manager=callback_manager,
            transformations=transformations,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            **kwargs,
        )

    @property
    def embed_model(self) -> OmniModalEmbeddingBundle[KD, KQ]:
        return self._embed_model

    @property
    def vector_stores(self) -> Mapping[KD, BasePydanticVectorStore]:
        return self._vector_stores

    def as_retriever(self, **kwargs: Any):
        # NOTE: lazy import
        from .retriever import OmniModalVectorIndexRetriever

        return OmniModalVectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            callback_manager=self._callback_manager,
            object_map=self._object_map,
            **kwargs,
        )

    def as_query_engine(self, llm: Optional[LLMType] = None, **kwargs: Any):
        # NOTE: lazy import
        from llama_index.core.query_engine.omni_modal import OmniModalQueryEngine

        retriever = self.as_retriever(**kwargs)

        llm = llm or llm_from_settings_or_context(Settings, self._service_context)
        assert isinstance(llm, MultiModalLLM), f"Wrong LLM type: {type(llm)}"

        return OmniModalQueryEngine(
            retriever,
            multi_modal_llm=llm,
            **kwargs,
        )

    def as_chat_engine(
        self,
        chat_mode: ChatMode = ChatMode.BEST,
        llm: Optional[LLMType] = None,
        **kwargs: Any,
    ) -> BaseChatEngine:
        return super().as_chat_engine(chat_mode, llm, **kwargs)

    def _embed_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> Dict[str, List[float]]:
        id_to_embed_map: Dict[str, List[float]] = {}

        nodes_to_embed: List[BaseNode] = []
        for node in nodes:
            if node.embedding is None:
                nodes_to_embed.append(node)
            else:
                id_to_embed_map[node.node_id] = node.embedding

        new_nodes = self.embed_model.embed_nodes(
            nodes_to_embed, show_progress=show_progress
        )

        for node in new_nodes:
            assert node.embedding is not None
            id_to_embed_map[node.node_id] = node.embedding

        return id_to_embed_map

    async def _aembed_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> Dict[str, List[float]]:
        id_to_embed_map: Dict[str, List[float]] = {}

        nodes_to_embed: List[BaseNode] = []
        for node in nodes:
            if node.embedding is None:
                nodes_to_embed.append(node)
            else:
                id_to_embed_map[node.node_id] = node.embedding

        new_nodes = await self.embed_model.aembed_nodes(
            nodes_to_embed, show_progress=show_progress
        )

        for node in new_nodes:
            assert node.embedding is not None
            id_to_embed_map[node.node_id] = node.embedding

        return id_to_embed_map

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = self._embed_nodes(nodes, show_progress=show_progress)

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
        id_to_embed_map = await self._aembed_nodes(nodes, show_progress=show_progress)

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            results.append(result)

        return results

    async def _async_add_nodes_to_index(
        self,
        index_struct: MultiModelIndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        nodes_by_modality = self.embed_model.group_documents_by_modality(nodes)
        new_ids_by_modality: Dict[Modality, List[str]] = defaultdict(list)

        for modality, group in nodes_by_modality:
            embedded_nodes = await self._aget_node_with_embedding(
                group, show_progress=show_progress
            )
            new_node_ids = await self._vector_stores[modality.key].async_add(
                embedded_nodes, **insert_kwargs
            )
            new_ids_by_modality[modality] = new_node_ids

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for new_ids in new_ids_by_modality.values():
                for node, new_id in zip(nodes, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    def _add_nodes_to_index(
        self,
        index_struct: MultiModelIndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        nodes_by_modality = self.embed_model.group_documents_by_modality(nodes)
        new_ids_by_modality: Dict[Modality, List[str]] = defaultdict(list)

        for modality, group in nodes_by_modality:
            embedded_nodes = self._get_node_with_embedding(
                group, show_progress=show_progress
            )
            new_node_ids = self._vector_stores[modality.key].add(
                embedded_nodes, **insert_kwargs
            )
            new_ids_by_modality[modality] = new_node_ids

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for key, new_ids in new_ids_by_modality.items():
                new_ids = new_ids_by_modality[key]
                for node, new_id in zip(nodes, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    def _build_index_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [
                self._async_add_nodes_to_index(
                    index_struct,
                    nodes,
                    show_progress=self._show_progress,
                    **insert_kwargs,
                )
            ]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(
                index_struct,
                nodes,
                show_progress=self._show_progress,
                **insert_kwargs,
            )
        return index_struct

    def build_index_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> IndexDict:
        """Build the index from nodes.

        NOTE: Overrides BaseIndex.build_index_from_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        # raise an error if even one node has no content
        if any(
            node.get_content(metadata_mode=MetadataMode.EMBED) == "" for node in nodes
        ):
            raise ValueError(
                "Cannot build index from nodes with no content. "
                "Please ensure all nodes have content."
            )

        return self._build_index_from_nodes(nodes, **insert_kwargs)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes, **insert_kwargs)

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes.

        NOTE: overrides BaseIndex.insert_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        for node in nodes:
            if isinstance(node, IndexNode):
                try:
                    node.dict()
                except ValueError:
                    self._object_map[node.index_id] = node.obj
                    node.obj = None

        with self._callback_manager.as_trace("insert_nodes"):
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
            node_ids (List[str]): A list of node_ids from the nodes to delete

        """
        raise NotImplementedError(
            "Vector indices currently only support delete_ref_doc, which "
            "deletes nodes using the ref_doc_id of ingested documents."
        )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        # delete from each tracked vector store
        for vector_store in self._vector_stores.values():
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
