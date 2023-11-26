"""Llama Dataset Class."""

import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Type

from dataclasses_json import DataClassJsonMixin
from pandas import DataFrame as PandasDataFrame

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.core import BaseQueryEngine


class CreatedByType(str, Enum):
    """The kinds of rag data examples."""

    HUMAN = "human"
    AI = "ai"

    def __str__(self) -> str:
        return self.value


@dataclass(repr=True)
class BaseLlamaPrediction(DataClassJsonMixin):
    """Base llama dataset example class."""

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Class name."""
        return "BaseLlamaPrediction"


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
    _predictions_type: Type[BaseLlamaPrediction]
    train_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )
    test_examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )
    train_predictions: Optional[List[BaseLlamaPrediction]] = Field(
        default=None, description="Predictions on the train examples."
    )
    test_predictions: Optional[List[BaseLlamaPrediction]] = Field(
        default=None, description="Predictions on the test examples."
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
            data = {
                "train_examples": train_examples,
                "test_examples": test_examples,
            }

            if self.train_predictions:
                train_predictions = [
                    self._predictions_type.to_dict(el) for el in self.train_predictions
                ]
                data.update({"train_predictions": train_predictions})
            if self.test_predictions:
                test_predictions = [
                    self._predictions_type.to_dict(el) for el in self.test_predictions
                ]
                data.update({"test_predictions": test_predictions})

            json.dump(data, f, indent=4)

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

        train_predictions, test_predictions = None, None
        if "train_predictions" in data:
            train_predictions = [
                cls._predictions_type.from_dict(el) for el in data["train_predictions"]
            ]
        if "test_predictions" in data:
            test_predictions = [
                cls._predictions_type.from_dict(el) for el in data["test_predictions"]
            ]

        return cls(
            train_examples=train_examples,
            test_examples=test_examples,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
        )

    @abstractmethod
    def _predict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaPrediction:
        """Subclasses need to generated this."""

    def _predict_examples(
        self, query_engine, on: Literal["train", "test"]
    ) -> List[BaseLlamaPrediction]:
        """Predictions on train examples."""
        if on == "train":
            examples = self.train_examples
        else:
            examples = self.test_examples

        predictions: List[BaseLlamaPrediction] = []
        for example in examples:
            prediction = self._predict_example(query_engine, example)
            predictions.append(prediction)
        return predictions

    def predict(
        self,
        query_engine: BaseQueryEngine,
        on_train: bool = True,
        on_test: bool = True,
        on_both: bool = True,
    ) -> None:
        """Predict with a given query engine."""
        train_predictions = None
        test_predictions = None

        if on_both:
            train_predictions = self._predict_examples(query_engine, on="train")
            test_predictions = self._predict_examples(query_engine, on="test")
        else:
            if on_train:
                train_predictions = self._predict_examples(query_engine, on="train")
            if on_test:
                test_predictions = self._predict_examples(query_engine, on="test")

        self.train_predictions = train_predictions
        self.test_predictions = test_predictions
