"""Llama Dataset Class."""

import asyncio
import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type

from dataclasses_json import DataClassJsonMixin
from pandas import DataFrame as PandasDataFrame

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.core import BaseQueryEngine
from llama_index.evaluation.eval_utils import asyncio_module


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

    def _predict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaExamplePrediction:
        """Sync version of _apredict_example."""
        return asyncio.run(
            self._apredict_example(query_engine=query_engine, example=example)
        )

    @abstractmethod
    async def _apredict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaExamplePrediction:
        """Subclasses need to generated this."""

    def _predict_examples(
        self, query_engine, show_progress: bool = False
    ) -> List[BaseLlamaExamplePrediction]:
        """Predictions on examples."""
        return asyncio.run(
            self._apredict_examples(
                query_engine=query_engine, show_progress=show_progress
            )
        )

    async def _apredict_examples(
        self, query_engine, show_progress: bool = False
    ) -> List[BaseLlamaExamplePrediction]:
        """Async predict on examples."""
        tasks = []
        for example in self.examples:
            tasks.append(self._apredict_example(query_engine, example))
        asyncio_mod = asyncio_module(show_progress=show_progress)
        return await asyncio_mod.gather(*tasks)

    @abstractmethod
    def _construct_prediction_dataset(self, predictions) -> BaseLlamaPredictionDataset:
        """Construct prediction dataset."""

    def make_predictions_with(
        self, query_engine: BaseQueryEngine, show_progress: bool = False
    ) -> BaseLlamaPredictionDataset:
        """Predict with a given query engine."""
        predictions = self._predict_examples(query_engine)
        return self._construct_prediction_dataset(predictions=predictions)

    async def amake_predictions_with(
        self, query_engine: BaseQueryEngine, show_progress: bool = False
    ) -> BaseLlamaPredictionDataset:
        """Predict with a given query engine."""
        predictions = await self._apredict_examples(query_engine)
        return self._construct_prediction_dataset(predictions=predictions)
