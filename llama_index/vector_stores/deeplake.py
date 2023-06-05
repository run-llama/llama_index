"""DeepLake vector store index.

An index that is built within DeepLake.

"""
import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np

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

    def __init__(
        self,
        dataset_path: str = "llama_index",
        token: Optional[str] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        ingestion_num_workers: int = 4,
        overwrite: bool = False,
    ):
        """Initialize with Deep Lake client."""
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = ingestion_num_workers
        self.token = token
        self.read_only = read_only
        self.dataset_path = dataset_path

        try:
            import deeplake
            from deeplake.constants import MB
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        self._deeplake = deeplake

        if deeplake.exists(dataset_path, token=token) and not overwrite:
            self.ds = deeplake.load(dataset_path, token=token, read_only=read_only)
            logger.warning(
                f"Deep Lake Dataset in {dataset_path} already exists, "
                f"loading from the storage"
            )
            self.ds.summary()
        else:
            self.ds = deeplake.empty(dataset_path, token=token, overwrite=True)

            with self.ds:
                self.ds.create_tensor(
                    "text",
                    htype="text",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )
                self.ds.create_tensor(
                    "metadata",
                    htype="json",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )
                self.ds.create_tensor(
                    "embedding",
                    htype="generic",
                    dtype=np.float64,
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    max_chunk_size=64 * MB,
                    create_shape_tensor=True,
                )
                self.ds.create_tensor(
                    "ids",
                    htype="text",
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    chunk_compression="lz4",
                )

    @property
    def client(self) -> None:
        """Get client."""
        return self.ds

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
        data_to_injest = []
        ids = []

        for result in embedding_results:
            embedding = result.embedding
            extra_info = result.node.extra_info or {}
            metadata = {**extra_info, **{"document_id": result.ref_doc_id}}
            id = result.id
            text = result.node.get_text()

            data_to_injest.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "ids": id,
                    "embedding": embedding,
                }
            )
            ids.append(id)

        @self._deeplake.compute
        def ingest(sample_in: List[Dict], sample_out: Any) -> None:
            for item in sample_in:
                sample_out.text.append(item["text"])
                sample_out.metadata.append(item["metadata"])
                sample_out.embedding.append(item["embedding"])
                sample_out.ids.append(item["ids"])

        batch_size = min(self.ingestion_batch_size, len(data_to_injest))
        batched = [
            data_to_injest[i : i + batch_size]
            for i in range(0, len(data_to_injest), batch_size)
        ]

        ingest().eval(
            batched,
            self.ds,
            num_workers=min(self.num_workers, len(batched) // self.num_workers),
        )

        self.ds.commit(allow_empty=True)
        self.ds.summary()
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        view = None
        if ref_doc_id:
            view = self.ds.filter(lambda x: x["ids"].numpy().tolist() == [ref_doc_id])
            ids = list(view.sample_indices)

        with self.ds:
            for id in sorted(ids)[::-1]:
                self.ds.pop(id)

            self.ds.commit(f"deleted {len(ids)} samples", allow_empty=True)
            self.ds.summary()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for DeepLake yet.")

        query_embedding = cast(List[float], query.query_embedding)
        embeddings = self.ds.embedding.numpy(fetch_chunks=True)
        embedding_ids = self.ds.ids.numpy(fetch_chunks=True)
        embedding_ids = [str(embedding_id[0]) for embedding_id in embedding_ids]

        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=embedding_ids,
        )

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
