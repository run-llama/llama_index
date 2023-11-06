"""DeepLake vector store index.

An index that is built within DeepLake.

"""
import logging
from typing import Any, List, Optional, cast

from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import VectorStore as VectorStoreBase
from llama_index.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

try:
    from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore

    DEEPLAKE_INSTALLED = True
except ImportError:
    DEEPLAKE_INSTALLED = False

logger = logging.getLogger(__name__)


class DeepLakeVectorStore(VectorStoreBase):
    """The DeepLake Vector Store.

    In this vector store we store the text, its embedding and
    a few pieces of its metadata in a deeplake dataset. This implemnetation
    allows the use of an already existing deeplake dataset if it is one that was created
    this vector store. It also supports creating a new one if the dataset doesn't
    exist or if `overwrite` is set to True.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        dataset_path: str = "llama_index",
        token: Optional[str] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        ingestion_num_workers: int = 4,
        overwrite: bool = False,
        exec_option: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            dataset_path (str): Path to the deeplake dataset, where data will be
            stored. Defaults to "llama_index".
            overwrite (bool, optional): Whether to overwrite existing dataset with same
                name. Defaults to False.
            token (str, optional): the deeplake token that allows you to access the
                dataset with proper access. Defaults to None.
            read_only (bool, optional): Whether to open the dataset with read only mode.
            ingestion_batch_size (int): used for controlling batched data
                injestion to deeplake dataset. Defaults to 1024.
            ingestion_num_workers (int): number of workers to use during data injestion.
                Defaults to 4.
            overwrite (bool): Whether to overwrite existing dataset with the
                new dataset with the same name.
            exec_option (str): Default method for search execution. It could be either
                It could be either ``"python"``, ``"compute_engine"`` or
                ``"tensor_db"``. Defaults to ``"python"``.
                - ``python`` - Pure-python implementation that runs on the client and
                    can be used for data stored anywhere. WARNING: using this option
                    with big datasets is discouraged because it can lead to memory
                    issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake
                    Compute Engine that runs on the client and can be used for any data
                    stored in or connected to Deep Lake. It cannot be used with
                    in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database
                    that is responsible for storage and query execution. Only available
                    for data stored in the Deep Lake Managed Database. Store datasets in
                    this database by specifying runtime = {"tensor_db": True} during
                    dataset creation.
            verbose (bool): Specify if verbose output is enabled. Default is True.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            ImportError: Unable to import `deeplake`.
        """
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = ingestion_num_workers
        self.token = token
        self.read_only = read_only
        self.dataset_path = dataset_path

        if not DEEPLAKE_INSTALLED:
            raise ImportError(
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
        self._id_tensor_name = "ids" if "ids" in self.vectorstore.tensors() else "id"

    @property
    def client(self) -> Any:
        """Get client.

        Returns:
            Any: DeepLake vectorstore dataset.
        """
        return self.vectorstore.dataset

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add the embeddings and their nodes into DeepLake.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Returns:
            List[str]: List of ids inserted.
        """
        embedding = []
        metadata = []
        id_ = []
        text = []

        for node in nodes:
            embedding.append(node.get_embedding())
            metadata.append(
                node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )
            )
            id_.append(node.node_id)
            text.append(node.get_content(metadata_mode=MetadataMode.NONE))

        kwargs = {
            "embedding": embedding,
            "metadata": metadata,
            self._id_tensor_name: id_,
            "text": text,
        }

        return self.vectorstore.add(
            return_ids=True,
            **kwargs,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self.vectorstore.delete(filter={"metadata": {"doc_id": ref_doc_id}})

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): VectorStoreQuery class input, it has
                the following attributes:
                1. query_embedding (List[float]): query embedding
                2. similarity_top_k (int): top k most similar nodes
            deep_memory (bool): Whether to use deep memory for query execution.

        Returns:
            VectorStoreQueryResult
        """
        query_embedding = cast(List[float], query.query_embedding)
        exec_option = kwargs.get("exec_option")
        deep_memory = kwargs.get("deep_memory")
        data = self.vectorstore.search(
            embedding=query_embedding,
            exec_option=exec_option,
            k=query.similarity_top_k,
            filter=query.filters,
            deep_memory=deep_memory,
        )

        similarities = data["score"]
        ids = data[self._id_tensor_name]
        metadatas = data["metadata"]
        nodes = []
        for metadata in metadatas:
            nodes.append(metadata_dict_to_node(metadata))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
