from typing import Any, Dict, Tuple

import numpy as np


class MockFaissIndex:
    """Mock Faiss index."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize params."""
        self._index: Dict[int, np.ndarray] = {}

    @property
    def ntotal(self) -> int:
        """Get ntotal."""
        return len(self._index)

    def add(self, vecs: np.ndarray) -> None:
        """Add vectors to index."""
        for vec in vecs:
            new_id = len(self._index)
            self._index[new_id] = vec

    def reset(self) -> None:
        """Reset index."""
        self._index = {}

    def search(self, vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search index."""
        # assume query vec is of the form 1 x k
        # index_mat is n x k
        index_mat = np.array(list(self._index.values()))
        # compute distances
        distances = np.linalg.norm(index_mat - vec, axis=1)

        indices = np.argsort(distances)[:k]
        sorted_distances = distances[indices][:k]

        # return distances and indices
        return sorted_distances[np.newaxis, :], indices[np.newaxis, :]
