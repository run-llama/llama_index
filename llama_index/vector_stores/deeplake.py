"""DeepLake vector store index.

An index that is built within DeepLake.

"""
import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np

from llama_index.schema import MetadataMode
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


def dp_filter(x: dict, filter: Dict[str, str]) -> bool:
    """Filter helper function for Deep Lake"""
    metadata = x["metadata"].data()["value"]
    return all(k in metadata and v == metadata[k] for k, v in filter.items())


class DeepLakeVectorStore(VectorStore):
    """The DeepLake Vector Store.

    In this vector store we store the text, its embedding and
    a few pieces of its metadata in a deeplake dataset. This implemnetation
    allows the use of an already existing deeplake dataset if it is one that was created
    this vector store. It also supports creating a new one if the dataset doesnt
    exist or if `overwrite` is set to True.

    Args:
        deeplake_path (str, optional): Path to the deeplake dataset, where data will be
        stored. Defaults to "llama_index".
        overwrite (bool, optional): Whether to overwrite existing dataset with same
            name. Defaults to False.
        token (str, optional): the deeplake token that allows you to access the dataset
            with proper access. Defaults to None.
        read_only (bool, optional): Whether to open the dataset with read only mode.
        ingestion_batch_size (bool, 1024): used for controlling batched data
            injestion to deeplake dataset. Defaults to 1024.
        injestion_num_workers (int, 1): number of workers to use during data injestion.
            Defaults to 4.
        overwrite (bool, optional): Whether to overwrite existing dataset with the
            new dataset with the same name.

    Raises:
        ImportError: Unable to import `deeplake`.
        UserNotLoggedinException: When user is not logged in with credentials
            or token.
        TokenPermissionError: When dataset does not exist or user doesn't have
            enough permissions to modify the dataset.
        InvalidTokenException: If the specified token is invalid


    Returns:
        DeepLakeVectorstore: Vectorstore that supports add, delete, and query.
    """

    stores_text: bool = False
    flat_metadata: bool = False

    def __init__(
        self,
        dataset_path: str = "llama_index",
        token: Optional[str] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        ingestion_num_workers: int = 4,
        overwrite: bool = False,
        exec_option: str = "python",
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize with Deep Lake client."""
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = ingestion_num_workers
        self.token = token
        self.read_only = read_only
        self.dataset_path = dataset_path

        try:
            from deeplake.core.vectorstore import VectorStore
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )

        self.vectorstore = VectorStore(
            path=dataset_path,
            ingestion_batch_size=ingestion_batch_size,
            num_workers=ingestion_num_workers,
            token=token,
            read_only=read_only,
            exec_option=exec_option,
            overwrite=overwrite,
            verbose=verbose,
            **kwargs,
        )

    @property
    def client(self) -> None:
        """Get client."""
        return self.vectorstore.dataset

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add the embeddings and their nodes into DeepLake.

        Args:
            embedding_results (List[NodeWithEmbedding]): The embeddings and their data
                to insert.

        Raises:
            UserNotLoggedinException: When user is not logged in with credentials
                or token.
            TokenPermissionError: When dataset does not exist or user doesn't have
                enough permissions to modify the dataset.
            InvalidTokenException: If the specified token is invalid

        Returns:
            List[str]: List of ids inserted.
        """
        embedding = []
        metadata = []
        id = []
        text = []

        for result in embedding_results:
            embedding.append(result.embedding)
            _metadata = result.node.metadata or {}
            metadata.append({**_metadata, **{"document_id": result.ref_doc_id}})
            id.append(result.id)
            text.append(result.node.get_content(metadata_mode=MetadataMode.NONE))

        return self.vectorstore.add(
            embedding=embedding,
            metadata=metadata,
            id=id,
            text=text,
            return_ids=True,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self.vectorstore.delete(ids=[ref_doc_id])

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for DeepLake yet.")

        query_embedding = cast(List[float], query.query_embedding)
        exec_option = kwargs.get("exec_option", "python")
        data = self.vectorstore.search(
            embedding=query_embedding,
            exec_option=exec_option,
        )

        similarities = data["score"]
        ids = data["id"]
        return VectorStoreQueryResult(similarities=similarities, ids=ids)
