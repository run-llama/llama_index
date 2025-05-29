from enum import Enum
from typing import TYPE_CHECKING, Union, overload

import numpy as np

if TYPE_CHECKING:
    import torch


class Pooling(str, Enum):
    """Enum of possible pooling choices with pooling behaviors."""

    CLS = "cls"
    MEAN = "mean"
    LAST = "last"  # last token pooling

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if self == self.CLS:
            return self.cls_pooling(array)
        elif self == self.LAST:
            return self.last_pooling(array)
        return self.mean_pooling(array)

    @classmethod
    @overload
    def cls_pooling(cls, array: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    @overload
    # TODO: Remove this `type: ignore` after the false positive problem
    #  is addressed in mypy: https://github.com/python/mypy/issues/15683 .
    def cls_pooling(cls, array: "torch.Tensor") -> "torch.Tensor":  # type: ignore
        ...

    @classmethod
    def cls_pooling(
        cls, array: "Union[np.ndarray, torch.Tensor]"
    ) -> "Union[np.ndarray, torch.Tensor]":
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

    @classmethod
    @overload
    def last_pooling(cls, array: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    @overload
    # TODO: Remove this `type: ignore` after the false positive problem
    #  is addressed in mypy: https://github.com/python/mypy/issues/15683 .
    def last_pooling(cls, array: "torch.Tensor") -> "torch.Tensor":  # type: ignore
        ...

    @classmethod
    def last_pooling(
        cls, array: "Union[np.ndarray, torch.Tensor]"
    ) -> "Union[np.ndarray, torch.Tensor]":
        if len(array.shape) == 3:
            return array[:, -1]
        if len(array.shape) == 2:
            return array[-1]
        raise NotImplementedError(f"Unhandled shape {array.shape}.")
