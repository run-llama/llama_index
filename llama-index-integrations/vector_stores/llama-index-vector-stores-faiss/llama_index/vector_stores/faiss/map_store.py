"""
Faiss Vector store map index.

This wraps the base Faiss vector store and adds handling for
the Faiss IDMap and IDMap2 indexes. This allows for
update/delete functionality through node_id and ref_doc_id mapping.

"""
import numpy as np
from typing import Any, List, Optional, cast

from llama_index.core.schema import BaseNode
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

class FaissVectorMapStore(FaissVectorStore):
    # ref_doc_id_map is used to map the ref_doc_id to fiass index id
    _ref_doc_id_map = PrivateAttr()
    # node_id_map is used to map the faiss index id to llama_index node id
    _node_id_map = PrivateAttr()

    def __init__(
        self,
        faiss_index: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `faiss` package not found. For instructions on
            how to install `faiss` please visit
            https://github.com/facebookresearch/faiss/wiki/Installing-Faiss
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(import_err_msg)

        if not isinstance(faiss_index, faiss.IndexIDMap) and not isinstance(faiss_index, faiss.IndexIDMap2):
            raise ValueError(
                "FaissVectorMapStore requires a faiss.IndexIDMap or faiss.IndexIDMap2 index. "
                "Please create an IndexIDMap2 index and pass it to the FaissVectorMapStore."
            )
        super().__init__(faiss_index=faiss_index)
        self._ref_doc_id_map = {}
        self._node_id_map = {}

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        NOTE: in the Faiss vector store, we do not store text in Faiss.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        new_ids = []
        for node in nodes:
            text_embedding = node.get_embedding()
            text_embedding_np = np.array(text_embedding, dtype="float32")[np.newaxis, :]
            new_id = str(self._faiss_index.ntotal)
            self._ref_doc_id_map[node.ref_doc_id] = self._faiss_index.ntotal
            self._node_id_map[new_id] = node.id_
            self._faiss_index.add_with_ids(text_embedding_np, self._faiss_index.ntotal)
            new_ids.append(node.id_)
        return new_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if ref_doc_id in self._ref_doc_id_map:
            # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#removing-elements-from-an-index
            self._faiss_index.remove_ids(np.array([int(self._ref_doc_id_map[ref_doc_id])], dtype=np.int64))
            del self._ref_doc_id_map[ref_doc_id]
        # node_id_map is only used for indext_struct handling and shouldn't reference an actual document
        if ref_doc_id in self._node_id_map:
            del self._node_id_map[ref_doc_id]

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from vector store."""
        faiss_ids = []
        for node_id in node_ids:
            # get the faiss id from the node_id_map
            faiss_id = self._node_id_map.get(node_id)
            if faiss_id is not None:
                faiss_ids.append(faiss_id)
        if not faiss_ids:
            return

        self._faiss_index.remove_ids(np.array(faiss_ids, dtype=np.int64))
        for node_id in node_ids:
            # get the faiss id from the node_id_map
            faiss_id = self._node_id_map.get(node_id)
            if faiss_id is not None:
                del self._node_id_map[node_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Faiss yet.")

        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = self._faiss_index.search(
            query_embedding_np, query.similarity_top_k
        )
        dists = list(dists[0])
        # if empty, then return an empty response
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])

        # returned dimension is 1 x k
        node_idxs = indices[0]

        filtered_dists = []
        filtered_node_idxs = []
        for dist, idx in zip(dists, node_idxs):
            if idx < 0:
                continue
            filtered_dists.append(dist)
            filtered_node_idxs.append(self._node_id_map[str(idx)])

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )
