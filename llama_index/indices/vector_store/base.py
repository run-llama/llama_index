"""Base vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.async_utils import run_async_tasks
from llama_index.data_structs.data_structs import IndexDict
from llama_index.data_structs.node import ImageNode, IndexNode, Node
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStore


class GPTVectorStoreIndex(BaseGPTIndex[IndexDict]):
    """Base GPT Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._use_async = use_async
        self._store_nodes_override = store_nodes_override
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            **kwargs,
        )

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

        return VectorIndexRetriever(
            self, doc_ids=list(self.index_struct.nodes_dict.values()), **kwargs
        )

    def _get_node_embedding_results(
        self, nodes: Sequence[Node]
    ) -> List[NodeWithEmbedding]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map: Dict[str, List[float]] = {}

        for n in nodes:
            if n.embedding is None:
                self._service_context.embed_model.queue_text_for_embedding(
                    n.get_doc_id(), n.get_text()
                )
            else:
                id_to_embed_map[n.get_doc_id()] = n.embedding

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = self._service_context.embed_model.get_queued_text_embeddings()
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.get_doc_id()]
            result = NodeWithEmbedding(node=node, embedding=embedding)
            results.append(result)
        return results

    async def _aget_node_embedding_results(
        self,
        nodes: Sequence[Node],
    ) -> List[NodeWithEmbedding]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map: Dict[str, List[float]] = {}

        text_queue: List[Tuple[str, str]] = []
        for n in nodes:
            if n.embedding is None:
                text_queue.append((n.get_doc_id(), n.get_text()))
            else:
                id_to_embed_map[n.get_doc_id()] = n.embedding

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = await self._service_context.embed_model.aget_queued_text_embeddings(
            text_queue
        )

        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.get_doc_id()]
            result = NodeWithEmbedding(node=node, embedding=embedding)
            results.append(result)
        return results

    async def _async_add_nodes_to_index(
        self, index_struct: IndexDict, nodes: Sequence[Node]
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        embedding_results = await self._aget_node_embedding_results(nodes)
        new_ids = self._vector_store.add(embedding_results)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)
                self._docstore.add_documents([result.node], allow_update=True)
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for result, new_id in zip(embedding_results, new_ids):
                if isinstance(result.node, (ImageNode, IndexNode)):
                    index_struct.add_node(result.node, text_id=new_id)
                    self._docstore.add_documents([result.node], allow_update=True)

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[Node],
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        embedding_results = self._get_node_embedding_results(nodes)
        new_ids = self._vector_store.add(embedding_results)

        if not self._vector_store.stores_text or self._store_nodes_override:
            # NOTE: if the vector store doesn't store text,
            # we need to add the nodes to the index struct and document store
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)
                self._docstore.add_documents([result.node], allow_update=True)
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for result, new_id in zip(embedding_results, new_ids):
                if isinstance(result.node, (ImageNode, IndexNode)):
                    index_struct.add_node(result.node, text_id=new_id)
                    self._docstore.add_documents([result.node], allow_update=True)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [self._async_add_nodes_to_index(index_struct, nodes)]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(index_struct, nodes)
        return index_struct

    @llm_token_counter("build_index_from_nodes")
    def build_index_from_nodes(self, nodes: Sequence[Node]) -> IndexDict:
        """Build the index from nodes.

        NOTE: Overrides BaseGPTIndex.build_index_from_nodes.
            GPTVectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        return self._build_index_from_nodes(nodes)

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes)

    @llm_token_counter("insert")
    def insert_nodes(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert nodes.

        NOTE: overrides BaseGPTIndex.insert_nodes.
            GPTVectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        self._insert(nodes, **insert_kwargs)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        self._index_struct.delete(doc_id)
        self._vector_store.delete(doc_id)
