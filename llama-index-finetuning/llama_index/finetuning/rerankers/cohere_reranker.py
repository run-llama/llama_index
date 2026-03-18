"""Cohere Reranker Finetuning Engine."""

import importlib.util
import os
import time
from typing import Optional

from cohere.finetuning import (  # pyright: ignore[reportMissingImports]
    BaseModel,
    FinetunedModel,
    Settings,
)

from llama_index.finetuning.types import BaseCohereRerankerFinetuningEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank  # pyright: ignore[reportMissingImports]


class CohereRerankerFinetuneEngine(BaseCohereRerankerFinetuningEngine):
    """Cohere Reranker Finetune Engine."""

    def __init__(
        self,
        train_file_name: str = "train.jsonl",
        val_file_name: Optional[str] = None,
        model_name: str = "exp_finetune",
        model_type: str = "RERANK",
        base_model: str = "english",
        api_key: Optional[str] = None,
    ) -> None:
        """Init params."""
        # This will be None if 'cohere' module is not available
        cohere_spec = importlib.util.find_spec("cohere")

        if cohere_spec is not None:
            import cohere  # pyright: ignore[reportMissingImports]
        else:
            # Raise an ImportError if 'cohere' is not installed
            raise ImportError(
                "Cannot import cohere. Please install the package using `pip install cohere`."
            )

        try:
            self.api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        self._model = cohere.Client(self.api_key, client_name="llama_index")
        self._train_file_name = train_file_name
        self._val_file_name = val_file_name
        self._model_name = model_name
        self._model_type = model_type
        self._base_model = base_model
        self._finetune_model = None

    def finetune(self) -> None:
        """Finetune model."""
        dataset_kwargs = {
            "name": self._model_name,
            "type": "rerank-finetune-input",
            "data": open(self._train_file_name, "rb"),
        }
        if self._val_file_name:
            dataset_kwargs["eval_data"] = open(self._val_file_name, "rb")

        dataset = self._model.datasets.create(**dataset_kwargs)

        while True:
            current_dataset = self._model.datasets.get(dataset.dataset.id)
            validation_status = current_dataset.dataset.validation_status
            if validation_status == "validated":
                break
            if validation_status == "failed":
                raise ValueError("Cohere dataset validation failed.")
            time.sleep(5)

        self._finetune_model = self._model.finetuning.create_finetuned_model(
            request=FinetunedModel(
                name=self._model_name,
                settings=Settings(
                    dataset_id=dataset.dataset.id,
                    base_model=BaseModel(
                        base_type="BASE_TYPE_RERANK",
                        name=self._base_model,
                    ),
                ),
            )
        )

    def get_finetuned_model(self, top_n: int = 5) -> CohereRerank:
        """Gets finetuned model id."""
        if self._finetune_model is None:
            raise RuntimeError(
                "Finetuned model is not set yet. Please run the finetune method first."
            )
        return CohereRerank(
            model=self._finetune_model.id, top_n=top_n, api_key=self.api_key
        )
