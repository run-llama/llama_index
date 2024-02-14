"""txtai Vector store index.

An index that is built on top of an existing vector store.

"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, List, Optional, cast

import fsspec
import numpy as np
from fsspec.implementations.local import LocalFileSystem

from llama_index.legacy.bridge.pydantic import PrivateAttr
from llama_index.legacy.schema import BaseNode
from llama_index.legacy.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.legacy.vector_stores.types import (
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
IMPORT_ERROR_MSG = """
    `txtai` package not found. For instructions on
    how to install `txtai` please visit
    https://neuml.github.io/txtai/install/
"""


class TxtaiVectorStore(BasePydanticVectorStore):
    """txtai Vector Store.

    Embeddings are stored within a txtai index.

    During query time, the index uses txtai to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        txtai_index (txtai.ann.ANN): txtai index instance

    """

    stores_text: bool = False

    _txtai_index = PrivateAttr()

    def __init__(
        self,
        txtai_index: Any,
    ) -> None:
        """Initialize params."""
        try:
            import txtai
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._txtai_index = cast(txtai.ann.ANN, txtai_index)

        super().__init__()

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "TxtaiVectorStore":
        persist_path = os.path.join(
            persist_dir,
            f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )
        # only support local storage for now
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("txtai only supports local storage for now.")
        return cls.from_persist_path(persist_path=persist_path, fs=None)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "TxtaiVectorStore":
        try:
            import txtai
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("txtai only supports local storage for now.")

        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")

        logger.info(f"Loading {__name__} config from {persist_path}.")
        parent_directory = Path(persist_path).parent
        config_path = parent_directory / "config.json"
        jsonconfig = config_path.exists()
        # Determine if config is json or pickle
        config_path = config_path if jsonconfig else parent_directory / "config"
        # Load configuration
        with open(config_path, "r" if jsonconfig else "rb") as f:
            config = json.load(f) if jsonconfig else pickle.load(f)

        logger.info(f"Loading {__name__} from {persist_path}.")
        txtai_index = txtai.ann.ANNFactory.create(config)
        txtai_index.load(persist_path)
        return cls(txtai_index=txtai_index)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        text_embedding_np = np.array(
            [node.get_embedding() for node in nodes], dtype="float32"
        )

        # Check if the ann index is already created
        # If not create the index with node embeddings
        if self._txtai_index.backend is None:
            self._txtai_index.index(text_embedding_np)
        else:
            self._txtai_index.append(text_embedding_np)

        indx_size = self._txtai_index.count()
        return [str(idx) for idx in range(indx_size - len(nodes) + 1, indx_size + 1)]

    @property
    def client(self) -> Any:
        """Return the txtai index."""
        return self._txtai_index

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
            raise NotImplementedError("txtai only supports local storage for now.")

        dirpath = Path(persist_path).parent
        dirpath.mkdir(exist_ok=True)

        jsonconfig = self._txtai_index.config.get("format", "pickle") == "json"
        # Determine if config is json or pickle
        config_path = dirpath / "config.json" if jsonconfig else dirpath / "config"

        # Write configuration
        with open(
            config_path,
            "w" if jsonconfig else "wb",
            encoding="utf-8" if jsonconfig else None,
        ) as f:
            if jsonconfig:
                # Write config as JSON
                json.dump(self._txtai_index.config, f, default=str)
            else:
                from txtai.version import __pickle__

                # Write config as pickle format
                pickle.dump(self._txtai_index.config, f, protocol=__pickle__)

        self._txtai_index.save(persist_path)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._txtai_index.delete([int(ref_doc_id)])

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query to search for in the index

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for txtai yet.")

        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        search_result = self._txtai_index.search(
            query_embedding_np, query.similarity_top_k
        )[0]
        # if empty, then return an empty response
        if len(search_result) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])

        filtered_dists = []
        filtered_node_idxs = []
        for dist, idx in search_result:
            if idx < 0:
                continue
            filtered_dists.append(dist)
            filtered_node_idxs.append(str(idx))

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )
