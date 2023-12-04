"""Llama Dataset Class."""

import json
from abc import abstractmethod
from enum import Enum
from typing import List, Optional, Type, Union

import tqdm
from pandas import DataFrame as PandasDataFrame

from llama_index.async_utils import asyncio_module
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.core import BaseQueryEngine


class CreatedByType(str, Enum):
    """The kinds of rag data examples."""

    HUMAN = "human"
    AI = "ai"

    def __str__(self) -> str:
        return self.value


class CreatedBy(BaseModel):
    model_name: Optional[str] = Field(
        default_factory=str, description="When CreatedByType.AI, specify model name."
    )
    type: CreatedByType

    def __str__(self) -> str:
        if self.type == "ai":
            return f"{self.type!s} ({self.model_name})"
        else:
            return str(self.type)


class BaseLlamaExamplePrediction(BaseModel):
    """Base llama dataset example class."""

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Class name."""
        return "BaseLlamaPrediction"


class BaseLlamaDataExample(BaseModel):
    """Base llama dataset example class."""

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Class name."""
        return "BaseLlamaDataExample"


class BaseLlamaPredictionDataset(BaseModel):
    _prediction_type: Type[BaseLlamaExamplePrediction] = BaseLlamaExamplePrediction  # type: ignore[misc]
    predictions: List[BaseLlamaExamplePrediction] = Field(
        default=list, description="Predictions on train_examples."
    )

    def __getitem__(self, val: Union[slice, int]) -> List[BaseLlamaExamplePrediction]:
        """Enable slicing and indexing.

        Returns the desired slice on `predictions`.
        """
        return self.predictions[val]

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            predictions = None
            if self.predictions:
                predictions = [
                    self._prediction_type.dict(el) for el in self.predictions
                ]
            data = {
                "predictions": predictions,
            }

            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "BaseLlamaPredictionDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        predictions = [cls._prediction_type.parse_obj(el) for el in data["predictions"]]

        return cls(
            predictions=predictions,
        )


class BaseLlamaDataset(BaseModel):
    _example_type: Type[BaseLlamaDataExample] = BaseLlamaDataExample  # type: ignore[misc]
    examples: List[BaseLlamaDataExample] = Field(
        default=[], description="Data examples of this dataset."
    )

    def __getitem__(self, val: Union[slice, int]) -> List[BaseLlamaDataExample]:
        """Enable slicing and indexing.

        Returns the desired slice on `examples`.
        """
        return self.examples[val]

    @abstractmethod
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            examples = [self._example_type.dict(el) for el in self.examples]
            data = {
                "examples": examples,
            }

            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "BaseLlamaDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        examples = [cls._example_type.parse_obj(el) for el in data["examples"]]

        return cls(
            examples=examples,
        )

    @abstractmethod
    def _construct_prediction_dataset(
        self, predictions: List[BaseLlamaExamplePrediction]
    ) -> BaseLlamaPredictionDataset:
        """Construct the specific prediction dataset.

        Args:
            predictions (List[BaseLlamaExamplePrediction]): the list of predictions.

        Returns:
            BaseLlamaPredictionDataset: A dataset of predictions.
        """

    @abstractmethod
    def _predict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaExamplePrediction:
        """Predict on a single example.

        NOTE: Subclasses need to implement this.

        Args:
            query_engine (BaseQueryEngine): Query engine to make the prediciton with.
            example (BaseLlamaDataExample): The example to predict on.

        Returns:
            BaseLlamaExamplePrediction: The prediction.
        """

    def make_predictions_with(
        self, query_engine: BaseQueryEngine, show_progress: bool = False
    ) -> BaseLlamaPredictionDataset:
        """Predict with a given query engine.

        Args:
            query_engine (BaseQueryEngine): The query engine to make predictions with.
            show_progress (bool, optional): Show progress of making predictions.

        Returns:
            BaseLlamaPredictionDataset: A dataset of predictions.
        """
        predictions = []
        if show_progress:
            example_iterator = tqdm.tqdm(self.examples)
        else:
            example_iterator = self.examples
        for example in example_iterator:
            predictions.append(self._predict_example(query_engine, example))
        return self._construct_prediction_dataset(predictions=predictions)

    # async methods
    @abstractmethod
    async def _apredict_example(
        self, query_engine: BaseQueryEngine, example: BaseLlamaDataExample
    ) -> BaseLlamaExamplePrediction:
        """Async predict on a single example.

        NOTE: Subclasses need to implement this.

        Args:
            query_engine (BaseQueryEngine): Query engine to make the prediciton with.
            example (BaseLlamaDataExample): The example to predict on.

        Returns:
            BaseLlamaExamplePrediction: The prediction.
        """

    async def amake_predictions_with(
        self, query_engine: BaseQueryEngine, show_progress: bool = False
    ) -> BaseLlamaPredictionDataset:
        """Async predict with a given query engine.

        Args:
            query_engine (BaseQueryEngine): The query engine to make predictions with.
            show_progress (bool, optional): Show progress of making predictions.

        Returns:
            BaseLlamaPredictionDataset: A dataset of predictions.
        """
        tasks = []
        for example in self.examples:
            tasks.append(self._apredict_example(query_engine, example))
        asyncio_mod = asyncio_module(show_progress=show_progress)
        predictions = await asyncio_mod.gather(*tasks)
        return self._construct_prediction_dataset(predictions=predictions)
