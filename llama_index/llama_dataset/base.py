"""Llama Dataset Class."""

import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type

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
class BaseLlamaExamplePrediction(DataClassJsonMixin):
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


class BaseLlamaPredictionDataset(BaseModel):
    _prediction_type: Type[BaseLlamaExamplePrediction] = BaseLlamaExamplePrediction
    predictions: Optional[List[BaseLlamaExamplePrediction]] = Field(
        default=None, description="Predictions on train_examples."
    )

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            predictions = [self._prediction_type.to_dict(el) for el in self.predictions]
            data = {
                "predictions": predictions,
            }

            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "BaseLlamaPredictionDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        predictions = [cls._prediction_type.from_dict(el) for el in data["predictions"]]

        return cls(
            predictions=predictions,
        )


class BaseLlamaDataset(BaseModel):
    _example_type: Type[BaseLlamaDataExample] = BaseLlamaDataExample
    examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            examples = [self._example_type.to_dict(el) for el in self.examples]
            data = {
                "examples": examples,
            }

            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "BaseLlamaDataset[BaseLlamaDataExample]":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        examples = [cls._example_type.from_dict(el) for el in data["examples"]]

        return cls(
            examples=examples,
        )

    @abstractmethod
    def _predict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaExamplePrediction:
        """Subclasses need to generated this."""

    def _predict_examples(self, query_engine) -> List[BaseLlamaExamplePrediction]:
        """Predictions on train examples."""
        predictions: List[BaseLlamaExamplePrediction] = []
        for example in self.examples:
            prediction = self._predict_example(query_engine, example)
            predictions.append(prediction)
        return predictions

    @abstractmethod
    def _construct_prediction_dataset(self, predictions) -> BaseLlamaPredictionDataset:
        """Construct prediction dataset."""

    def make_predictions_with(
        self,
        query_engine: BaseQueryEngine,
    ) -> BaseLlamaPredictionDataset:
        """Predict with a given query engine."""
        predictions = self._predict_examples(query_engine)
        return self._construct_prediction_dataset(predictions=predictions)
