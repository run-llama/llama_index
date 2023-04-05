"""Base vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from gpt_index.async_utils import run_async_tasks
from gpt_index.constants import VECTOR_STORE_KEY
from gpt_index.data_structs.data_structs_v2 import IndexDict
from gpt_index.data_structs.node_v2 import ImageNode, IndexNode, Node
from gpt_index.indices.base import BaseGPTIndex, QueryMap
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.vector_store.base_query import GPTVectorStoreIndexQuery
from gpt_index.token_counter.token_counter import llm_token_counter
from gpt_index.vector_stores.registry import (
    load_vector_store_from_dict,
    save_vector_store_to_dict,
)
from gpt_index.vector_stores.simple import SimpleVectorStore
from gpt_index.vector_stores.types import NodeEmbeddingResult, VectorStore


class GPTVectorStoreIndex(BaseGPTIndex[IndexDict]):
    """Base GPT Vector Store Index.

    Args:
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        vector_store (Optional[VectorStore]): Vector store to use for
            embedding similarity. See :ref:`Ref-Indices-VectorStore-Stores`
            for more details.
        use_async (bool): Whether to use asynchronous calls. Defaults to False.

    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        vector_store: Optional[VectorStore] = None,
        use_async: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._vector_store = vector_store or SimpleVectorStore()

        self._use_async = use_async
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> QueryMap:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTVectorStoreIndexQuery,
            QueryMode.EMBEDDING: GPTVectorStoreIndexQuery,
        }

    def _get_node_embedding_results(
        self, nodes: Sequence[Node], existing_node_ids: Set
    ) -> List[NodeEmbeddingResult]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_node_map: Dict[str, Node] = {}
        id_to_embed_map: Dict[str, List[float]] = {}

        for n in nodes:
            new_id = n.get_doc_id()
            if n.embedding is None:
                self._service_context.embed_model.queue_text_for_embeddding(
                    new_id, n.get_text()
                )
            else:
                id_to_embed_map[new_id] = n.embedding

            id_to_node_map[new_id] = n

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = self._service_context.embed_model.get_queued_text_embeddings()
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        result_tups = []
        for id, embed in id_to_embed_map.items():
            doc_id = id_to_node_map[id].ref_doc_id
            if doc_id is None:
                raise ValueError("Reference doc id is None.")
            result_tups.append(
                NodeEmbeddingResult(id, id_to_node_map[id], embed, doc_id=doc_id)
            )
        return result_tups

    async def _aget_node_embedding_results(
        self,
        nodes: Sequence[Node],
        existing_node_ids: Set,
    ) -> List[NodeEmbeddingResult]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_node_map: Dict[str, Node] = {}
        id_to_embed_map: Dict[str, List[float]] = {}

        text_queue: List[Tuple[str, str]] = []
        for n in nodes:
            new_id = n.get_doc_id()
            if n.embedding is None:
                text_queue.append((new_id, n.get_text()))
            else:
                id_to_embed_map[new_id] = n.embedding

            id_to_node_map[new_id] = n

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = await self._service_context.embed_model.aget_queued_text_embeddings(
            text_queue
        )
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        result_tups = []
        for id, embed in id_to_embed_map.items():
            doc_id = id_to_node_map[id].ref_doc_id
            if doc_id is None:
                raise ValueError("Reference doc id is None.")
            result_tups.append(
                NodeEmbeddingResult(id, id_to_node_map[id], embed, doc_id=doc_id)
            )
        return result_tups

    async def _async_add_nodes_to_index(
        self, index_struct: IndexDict, nodes: Sequence[Node]
    ) -> None:
        """Asynchronously add nodes to index."""
        embedding_results = await self._aget_node_embedding_results(
            nodes,
            set(),
        )

        new_ids = self._vector_store.add(embedding_results)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text:
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)
                self._docstore.add_documents([result.node], allow_update=True)

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[Node],
    ) -> None:
        """Add document to index."""
        embedding_results = self._get_node_embedding_results(
            nodes,
            set(),
        )

        new_ids = self._vector_store.add(embedding_results)

        if not self._vector_store.stores_text:
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

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        self._index_struct.delete(doc_id)
        self._vector_store.delete(doc_id)

    @classmethod
    def load_from_dict(
        cls, result_dict: Dict[str, Any], **kwargs: Any
    ) -> "BaseGPTIndex":
        """Load index from string (in JSON-format).

        This method loads the index from a JSON string. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        NOTE: load_from_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Args:
            index_string (str): The index string (in JSON-format).

        Returns:
            BaseGPTIndex: The loaded index.

        """
        vector_store = load_vector_store_from_dict(
            result_dict[VECTOR_STORE_KEY], **kwargs
        )
        return super().load_from_dict(result_dict, vector_store=vector_store, **kwargs)

    def save_to_dict(self, **save_kwargs: Any) -> dict:
        """Save to string.

        This method stores the index into a JSON string.

        NOTE: save_to_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Returns:
            dict: The JSON dict of the index.

        """
        out_dict = super().save_to_dict()
        out_dict[VECTOR_STORE_KEY] = save_vector_store_to_dict(self._vector_store)
        return out_dict

    @property
    def query_context(self) -> Dict[str, Any]:
        return {"vector_store": self._vector_store}
