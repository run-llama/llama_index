"""Hnswlib Vector store index.

An index that is built on top of an existing vector store.

"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, cast, Literal

import fsspec
import numpy as np
from fsspec.implementations.local import LocalFileSystem

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
    NAMESPACE_SEP,
)
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger()

DEFAULT_PERSIST_PATH = os.path.join(
    DEFAULT_PERSIST_DIR,
    f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
)

IMPORT_ERROR_MSG = """
    `Hnswlib` package not found. For instructions on
    how to install `Hnswlib` please visit
    https://github.com/nmslib/hnswlib/
"""


class HnswlibVectorStore(BasePydanticVectorStore):
    """Hnswlib Vector Store.

    Embeddings are stored within a Hnswlib index.

    During query time, the index uses Hierarchical Navigable Small World
    approximate nearest neighbors algorithm to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        hnswlib_index (hnswlib.Index): Hnswlib index instance

    Examples:
        `pip install llama-index-vector-stores-hnswlib hnswlib`

        ```python
        from llama_index.vector_stores.hnswlib import HnswlibVectorStore
        import Hnswlib

        # create a Hnswlib index
        dim = 768  # dimension
        space = 'cosine' # distance function
        max_elements = 1000 # maximum number of elements that Hnswlib.Index can store. NOTE: Hnswlib.Index is resizeable

        hnswlib_index = hnswlib_index(space, dim)
        hnswlib_index.init_index(max_elements, **kwargs)
        vector_store = HnswlibVectorStore(hnswlib_index=hnswlib_index)

        # or

        vector_store = HnswlibVectorStore.from_params(space, dim, max_elements, **kwargs)

        ```
    """

    stores_text: bool = False
    _hnswlib_index = PrivateAttr()

    def __init__(self, hnswlib_index: Any) -> None:
        try:
            import hnswlib
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        super().__init__()
        self._hnswlib_index = cast(hnswlib.Index, hnswlib_index)

    @classmethod
    def from_params(
        cls,
        space: Literal["ip", "cosine", "l2"],
        dimension: int,
        max_elements: int,
        ef: Optional[int] = None,
        **kwargs,
    ) -> "HnswlibVectorStore":
        """To avoid creating the `Hnswlib.Index` Yourself You can just specify it's params.
        For more details see [Hnswlib documentation](https://github.com/nmslib/hnswlib?tab=readme-ov-file#api-description).
        """
        try:
            import hnswlib
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        hnswlib_index = hnswlib.Index(space, dimension)
        hnswlib_index.init_index(max_elements, **kwargs)
        if ef is not None:
            hnswlib_index.set_ef(ef)
        return cls(hnswlib_index)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "HnswlibVectorStore":
        persist_path = os.path.join(
            persist_dir, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
        )
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("Hnswlib only supports local storage for now.")
        return cls.from_persist_path(persist_path=persist_path, fs=None)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "HnswlibVectorStore":
        try:
            import hnswlib
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("Hnswlib only supports local storage for now.")

        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")
        parent_directory = Path(persist_path).parent
        config_path = parent_directory / "config.json"
        if not config_path.exists():
            raise ValueError(f"No existing config.json found at {config_path}")
        logger.info(f"Loading {__name__} from {persist_path}.")
        with open(config_path) as file:
            config = json.load(file)
        logger.info(f"Loading {__name__} from {persist_path}.")
        hnswlib_index = hnswlib.Index(
            space=config["space"],
            dim=config["dim"],
        )
        hnswlib_index.load_index(persist_path)
        hnswlib_index.set_ef(config["ef"])
        return cls(hnswlib_index=hnswlib_index)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        embeddings = np.array([node.get_embedding() for node in nodes])
        self._hnswlib_index.add_items(embeddings, **add_kwargs)
        index_size = self._hnswlib_index.get_current_count()
        return [str(id) for id in range(index_size - len(nodes), index_size)]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for Hnswlib index.")

    @property
    def client(self):
        """Return the Hnswlib index."""
        return self._hnswlib_index

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Save to file.

        This method saves the vector store to disk.

        Args:
            persist_path (str): The save_path of the file.

        """
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("Hnswlib only supports local storage for now.")

        parent_directory = Path(persist_path).parent
        parent_directory.mkdir(exist_ok=True)

        config_path = parent_directory / "config.json"
        with open(config_path, "w") as file:
            json.dump(
                {
                    "dim": self._hnswlib_index.dim,
                    "space": self._hnswlib_index.space,
                    "ef": self._hnswlib_index.ef,
                    "ef_construction": self._hnswlib_index.ef_construction,
                    "element_count": self._hnswlib_index.element_count,
                    "max_elements": self._hnswlib_index.max_elements,
                },
                file,
            )
        self._hnswlib_index.save_index(persist_path)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            ef (int): higher ef leads to better accuracy, but slower search

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Hnswlib yet.")
        if "ef" in kwargs:
            self._hnswlib_index.set_ef(kwargs["ef"])
        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        indices, distances = self._hnswlib_index.knn_query(
            query_embedding_np, query.similarity_top_k
        )
        node_idxs = indices.tolist()[0]
        distances = distances[0].tolist()
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])
        filtered_dists = []
        filtered_node_idxs = []
        for dist, idx in zip(distances, node_idxs):
            if idx < 0:
                continue
            filtered_dists.append(dist)
            filtered_node_idxs.append(str(idx))

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )
