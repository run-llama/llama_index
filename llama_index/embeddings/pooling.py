from enum import Enum

import numpy as np


class Pooling(str, Enum):
    """Enum of possible pooling choices with pooling behaviors."""

    CLS = "cls"
    MEAN = "mean"

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if self == self.CLS:
            return self.cls_pooling(array)
        return self.mean_pooling(array)

    @classmethod
    def cls_pooling(cls, array: np.ndarray) -> np.ndarray:
        if len(array.shape) == 3:
            return array[:, 0]
        if len(array.shape) == 2:
            return array[0]
        raise NotImplementedError(f"Unhandled shape {array.shape}.")

    @classmethod
    def mean_pooling(cls, array: np.ndarray) -> np.ndarray:
        if len(array.shape) == 3:
            return array.mean(axis=1)
        if len(array.shape) == 2:
            return array.mean(axis=0)
        raise NotImplementedError(f"Unhandled shape {array.shape}.")
