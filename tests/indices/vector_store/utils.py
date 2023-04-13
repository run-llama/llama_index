from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np


class MockPineconeIndex:
    def __init__(self) -> None:
        """Mock pinecone index."""
        self._tuples: List[Dict[str, Any]] = []

    def upsert(self, tuples: List[Dict[str, Any]], **kwargs: Any) -> None:
        """Mock upsert."""
        self._tuples.extend(tuples)

    def delete(self, ids: List[str]) -> None:
        """Mock delete."""
        new_tuples = []
        for tup in self._tuples:
            if tup["id"] not in ids:
                new_tuples.append(tup)
        self._tuples = new_tuples

    def query(
        self,
        vector: Optional[List[float]] = None,
        sparse_vector: Optional[List[float]] = None,
        top_k: int = 1,
        include_values: bool = True,
        include_metadata: bool = True,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> Any:
        """Mock query."""
        # index_mat is n x k
        index_mat = np.array([tup["values"] for tup in self._tuples])
        query_vec = np.array(vector)[np.newaxis, :]

        # compute distances
        distances = np.linalg.norm(index_mat - query_vec, axis=1)

        indices = np.argsort(distances)[:top_k]
        # sorted_distances = distances[indices][:top_k]

        matches = []
        for index in indices:
            tup = self._tuples[index]
            match = MagicMock()
            match.metadata = {
                "text": tup["metadata"]["text"],
                "doc_id": tup["metadata"]["doc_id"],
                "id": tup["metadata"]["id"],
            }

            matches.append(match)

        response = MagicMock()
        response.matches = matches
        return response
