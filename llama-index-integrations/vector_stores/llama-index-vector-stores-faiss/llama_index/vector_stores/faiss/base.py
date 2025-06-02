"""
Faiss Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, List, Optional, cast

import fsspec
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger()

DEFAULT_PERSIST_PATH = os.path.join(
    DEFAULT_PERSIST_DIR, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
)


class FaissVectorStore(BasePydanticVectorStore):
    """
    Faiss Vector Store.

    Embeddings are stored within a Faiss index.

    During query time, the index uses Faiss to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        faiss_index (faiss.Index): Faiss index instance

    Examples:
        `pip install llama-index-vector-stores-faiss faiss-cpu`

        ```python
        from llama_index.vector_stores.faiss import FaissVectorStore
        import faiss

        # create a faiss index
        d = 1536  # dimension
        faiss_index = faiss.IndexFlatL2(d)

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        ```

    """

    stores_text: bool = False

    _faiss_index = PrivateAttr()

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

        super().__init__()

        self._faiss_index = cast(faiss.Index, faiss_index)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "FaissVectorStore":
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
    ) -> "FaissVectorStore":
        import faiss

        # I don't think FAISS supports fsspec, it requires a path in the SWIG interface
        # TODO: copy to a temp file and load into memory from there
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("FAISS only supports local storage for now.")

        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")

        logger.info(f"Loading {__name__} from {persist_path}.")
        faiss_index = faiss.read_index(persist_path)
        return cls(faiss_index=faiss_index)

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
            self._faiss_index.add(text_embedding_np)
            new_ids.append(new_id)
        return new_ids

    @property
    def client(self) -> Any:
        """Return the faiss index."""
        return self._faiss_index

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
        # I don't think FAISS supports fsspec, it requires a path in the SWIG interface
        # TODO: write to a temporary file and then copy to the final destination
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("FAISS only supports local storage for now.")
        import faiss

        dirpath = os.path.dirname(persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        faiss.write_index(self._faiss_index, persist_path)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for Faiss index.")

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
            filtered_node_idxs.append(str(idx))

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )
