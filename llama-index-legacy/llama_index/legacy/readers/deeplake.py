"""DeepLake reader."""
from typing import List, Optional, Union

import numpy as np

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

distance_metric_map = {
    "l2": lambda a, b: np.linalg.norm(a - b, axis=1, ord=2),
    "l1": lambda a, b: np.linalg.norm(a - b, axis=1, ord=1),
    "max": lambda a, b: np.linalg.norm(a - b, axis=1, ord=np.inf),
    "cos": lambda a, b: np.dot(a, b.T)
    / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)),
    "dot": lambda a, b: np.dot(a, b.T),
}


def vector_search(
    query_vector: Union[List, np.ndarray],
    data_vectors: np.ndarray,
    distance_metric: str = "l2",
    limit: Optional[int] = 4,
) -> List:
    """Naive search for nearest neighbors
    args:
        query_vector: Union[List, np.ndarray]
        data_vectors: np.ndarray
        limit (int): number of nearest neighbors
        distance_metric: distance function 'L2' for Euclidean, 'L1' for Nuclear, 'Max'
            l-infinity distance, 'cos' for cosine similarity, 'dot' for dot product
    returns:
        nearest_indices: List, indices of nearest neighbors.
    """
    # Calculate the distance between the query_vector and all data_vectors
    if isinstance(query_vector, list):
        query_vector = np.array(query_vector)
        query_vector = query_vector.reshape(1, -1)

    distances = distance_metric_map[distance_metric](query_vector, data_vectors)
    nearest_indices = np.argsort(distances)

    nearest_indices = (
        nearest_indices[::-1][:limit]
        if distance_metric in ["cos"]
        else nearest_indices[:limit]
    )

    return nearest_indices.tolist()


class DeepLakeReader(BaseReader):
    """DeepLake reader.

    Retrieve documents from existing DeepLake datasets.

    Args:
        dataset_name: Name of the deeplake dataset.
    """

    def __init__(
        self,
        token: Optional[str] = None,
    ):
        """Initializing the deepLake reader."""
        import_err_msg = (
            "`deeplake` package not found, please run `pip install deeplake`"
        )
        try:
            import deeplake  # noqa
        except ImportError:
            raise ImportError(import_err_msg)
        self.token = token

    def load_data(
        self,
        query_vector: List[float],
        dataset_path: str,
        limit: int = 4,
        distance_metric: str = "l2",
    ) -> List[Document]:
        """Load data from DeepLake.

        Args:
            dataset_name (str): Name of the DeepLake dataset.
            query_vector (List[float]): Query vector.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.
        """
        import deeplake
        from deeplake.util.exceptions import TensorDoesNotExistError

        dataset = deeplake.load(dataset_path, token=self.token)

        try:
            embeddings = dataset.embedding.numpy(fetch_chunks=True)
        except Exception:
            raise TensorDoesNotExistError("embedding")

        indices = vector_search(
            query_vector, embeddings, distance_metric=distance_metric, limit=limit
        )

        documents = []
        for idx in indices:
            document = Document(
                text=str(dataset[idx].text.numpy().tolist()[0]),
                id_=dataset[idx].ids.numpy().tolist()[0],
            )

            documents.append(document)

        return documents
