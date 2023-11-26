"""Llama Dataset Class."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

from dataclasses_json import DataClassJsonMixin
from pandas import DataFrame as PandasDataFrame

from llama_index.bridge.pydantic import BaseModel, Field


@dataclass(repr=True)
class BaseLlamaDataExample(DataClassJsonMixin):
    """Base llama dataset example class."""

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Class name."""
        return "BaseLlamaDataExample"


class BaseLlamaDataset(BaseModel):
    train_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )
    test_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""


class CreatedByType(str, Enum):
    """The kinds of rag data examples."""

    HUMAN = "human"
    AI = "ai"

    def __str__(self) -> str:
        return self.value
