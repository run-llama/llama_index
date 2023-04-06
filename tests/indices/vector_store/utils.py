from typing import Any, Dict, List, Tuple, Optional
from unittest.mock import MagicMock

import numpy as np


class MockPineconeIndex:
    def __init__(self) -> None:
        """Mock pinecone index."""
        self._tuples: List[Tuple[str, List[float], Dict]] = []

    def upsert(
        self, tuples: List[Tuple[str, List[float], Dict]], **kwargs: Any
    ) -> None:
        """Mock upsert."""
        self._tuples.extend(tuples)

    def delete(self, ids: List[str]) -> None:
        """Mock delete."""
        new_tuples = []
        for tup in self._tuples:
            if tup[0] not in ids:
                new_tuples.append(tup)
        self._tuples = new_tuples

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        include_values: bool = True,
        include_metadata: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Mock query."""
        # index_mat is n x k
        index_mat = np.array([tup[1] for tup in self._tuples])
        query_vec = np.array(query_embedding)[np.newaxis, :]

        # compute distances
        distances = np.linalg.norm(index_mat - query_vec, axis=1)

        indices = np.argsort(distances)[:top_k]
        # sorted_distances = distances[indices][:top_k]

        matches = []
        for index in indices:
            tup = self._tuples[index]
            match = MagicMock()
            match.metadata = {
                "text": tup[2]["text"],
                "doc_id": tup[2]["doc_id"],
                "id": tup[2]["id"],
            }

            matches.append(match)

        response = MagicMock()
        response.matches = matches
        return response
