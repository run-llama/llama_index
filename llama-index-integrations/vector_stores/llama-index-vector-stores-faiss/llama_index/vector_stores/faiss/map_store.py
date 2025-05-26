"""
Faiss Map Vector Store index.

An index that is built on top of an existing vector store.

"""

import os
from typing import Any, List, Optional, cast

import numpy as np
import fsspec
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.schema import BaseNode
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.faiss.base import (
    FaissVectorStore,
    DEFAULT_PERSIST_PATH,
    DEFAULT_PERSIST_FNAME,
)
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

DEFAULT_ID_MAP_NAME = "id_map.json"


class FaissMapVectorStore(FaissVectorStore):
    """
    Faiss Map Vector Store.

    This wraps the base Faiss vector store and adds handling for
    the Faiss IDMap and IDMap2 indexes. This allows for
    update/delete functionality through node_id and faiss_id mapping.

    Embeddings are stored within a Faiss index.

    During query time, the index uses Faiss to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        faiss_index (faiss.IndexIDMap or faiss.IndexIDMap2): Faiss id map index instance

    Examples:
        `pip install llama-index-vector-stores-faiss faiss-cpu`

        ```python
        from llama_index.vector_stores.faiss import FaissMapVectorStore
        import faiss

        # create a faiss index
        d = 1536  # dimension
        faiss_index = faiss.IndexFlatL2(d)

        # wrap it in an IDMap or IDMap2
        id_map_index = faiss.IndexIDMap2(faiss_index)

        vector_store = FaissMapVectorStore(faiss_index=id_map_index)
        ```

    """

    # _node_id_to_faiss_id_map is used to map the node id to the faiss id
    _node_id_to_faiss_id_map = PrivateAttr()
    # _faiss_id_to_node_id_map is used to map the faiss id to the node id
    _faiss_id_to_node_id_map = PrivateAttr()

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

        if not isinstance(faiss_index, faiss.IndexIDMap) and not isinstance(
            faiss_index, faiss.IndexIDMap2
        ):
            raise ValueError(
                "FaissVectorMapStore requires a faiss.IndexIDMap or faiss.IndexIDMap2 index. "
                "Please create an IndexIDMap2 index and pass it to the FaissVectorMapStore."
            )
        super().__init__(faiss_index=faiss_index)
        self._node_id_to_faiss_id_map = {}
        self._faiss_id_to_node_id_map = {}

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
            self._node_id_to_faiss_id_map[node.id_] = self._faiss_index.ntotal
            self._faiss_id_to_node_id_map[self._faiss_index.ntotal] = node.id_
            self._faiss_index.add_with_ids(text_embedding_np, self._faiss_index.ntotal)
            new_ids.append(node.id_)
        return new_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # only handle delete on node_ids
        if ref_doc_id in self._node_id_to_faiss_id_map:
            faiss_id = self._node_id_to_faiss_id_map[ref_doc_id]
            # remove the faiss id from the faiss index
            self._faiss_index.remove_ids(np.array([faiss_id], dtype=np.int64))
            # remove the node id from the node id map
            if ref_doc_id in self._node_id_to_faiss_id_map:
                del self._node_id_to_faiss_id_map[ref_doc_id]
            # remove the faiss id from the faiss id map
            if faiss_id in self._faiss_id_to_node_id_map:
                del self._faiss_id_to_node_id_map[faiss_id]

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from vector store."""
        if filters is not None:
            raise NotImplementedError("Metadata filters not implemented for Faiss yet.")

        if node_ids is None:
            raise ValueError("node_ids must be provided to delete nodes.")

        faiss_ids = []
        for node_id in node_ids:
            # get the faiss id from the node_id_map
            faiss_id = self._node_id_to_faiss_id_map.get(node_id)
            if faiss_id is not None:
                faiss_ids.append(faiss_id)
        if not faiss_ids:
            return

        self._faiss_index.remove_ids(np.array(faiss_ids, dtype=np.int64))

        # cleanup references
        for node_id in node_ids:
            # get the faiss id from the node_id_map
            faiss_id = self._node_id_to_faiss_id_map.get(node_id)
            if faiss_id is not None and faiss_id in self._faiss_id_to_node_id_map:
                del self._faiss_id_to_node_id_map[faiss_id]
            if node_id in self._node_id_to_faiss_id_map:
                del self._node_id_to_faiss_id_map[node_id]

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
            filtered_node_idxs.append(self._faiss_id_to_node_id_map[idx])

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """
        Save to file.

        This method saves the vector store to disk.

        Args:
            persist_path (str): The save_path of the file.

        """
        super().persist(persist_path=persist_path, fs=fs)
        dirpath = os.path.dirname(persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        id_map = {}
        id_map["node_id_to_faiss_id_map"] = self._node_id_to_faiss_id_map
        id_map["faiss_id_to_node_id_map"] = self._faiss_id_to_node_id_map
        # save the id map
        id_map_path = os.path.join(dirpath, DEFAULT_ID_MAP_NAME)
        with open(id_map_path, "w") as f:
            f.write(str(id_map))

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "FaissMapVectorStore":
        persist_path = os.path.join(
            persist_dir,
            f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )
        # only support local storage for now
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("FAISS only supports local storage for now.")
        return cls.from_persist_path(persist_path=persist_path, fs=None)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "FaissMapVectorStore":
        import faiss

        # I don't think FAISS supports fsspec, it requires a path in the SWIG interface
        # TODO: copy to a temp file and load into memory from there
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("FAISS only supports local storage for now.")

        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")

        dirpath = os.path.dirname(persist_path)
        id_map_path = os.path.join(dirpath, DEFAULT_ID_MAP_NAME)
        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")

        faiss_index = faiss.read_index(persist_path)
        with open(id_map_path, "r") as f:
            id_map = eval(f.read())

        map_vs = cls(faiss_index=faiss_index)
        map_vs._node_id_to_faiss_id_map = id_map["node_id_to_faiss_id_map"]
        map_vs._faiss_id_to_node_id_map = id_map["faiss_id_to_node_id_map"]
        return map_vs
