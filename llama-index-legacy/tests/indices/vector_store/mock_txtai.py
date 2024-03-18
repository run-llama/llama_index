from typing import Any, Dict, List, Tuple

import numpy as np


class MockTxtaiIndex:
    """Mock txtai index."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize params."""
        self._index: Dict[int, np.ndarray] = {}
        self.backend = None

    def count(self) -> int:
        """Get count."""
        return len(self._index)

    def index(self, vecs: np.ndarray) -> None:
        """Index vectors to index."""
        self._index.clear()
        self.add(vecs)

    def add(self, vecs: np.ndarray) -> None:
        """Add vectors to index."""
        for vec in vecs:
            new_id = len(self._index)
            self._index[new_id] = vec

    def reset(self) -> None:
        """Reset index."""
        self._index = {}

    def search(self, vec: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        """Search index."""
        # assume query vec is of the form 1 x k
        # index_mat is n x k
        index_mat = np.array(list(self._index.values()))
        # compute distances
        scores = np.linalg.norm(index_mat - vec, axis=1)

        indices = np.argsort(scores)[:k]
        sorted_distances = scores[indices][:k]

        # return scores and indices
        return [list(zip(indices, sorted_distances))]
