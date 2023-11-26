"""Llama Dataset Class."""

import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Type

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
    _examples_type: Type[BaseLlamaDataExample]
    train_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )
    test_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            train_examples = [
                self._examples_type.to_dict(el) for el in self.train_examples
            ]
            test_examples = [
                self._examples_type.to_dict(el) for el in self.test_examples
            ]
            json.dump(
                {"train_examples": train_examples, "test_examples": test_examples},
                f,
                indent=4,
            )

    @classmethod
    def from_json(cls, path: str) -> "BaseLlamaDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)
        train_examples = [
            cls._examples_type.from_dict(el) for el in data["train_examples"]
        ]
        test_examples = [
            cls._examples_type.from_dict(el) for el in data["test_examples"]
        ]
        return cls(train_examples=train_examples, test_examples=test_examples)


class CreatedByType(str, Enum):
    """The kinds of rag data examples."""

    HUMAN = "human"
    AI = "ai"

    def __str__(self) -> str:
        return self.value
