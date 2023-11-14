"""Cohere Reranker Finetuning Engine."""
import importlib.util
from typing import Any

from llama_index.finetuning.types import BaseCohereRerankerFinetuningEngine


class CohereRerankerFinetuneEngine(BaseCohereRerankerFinetuningEngine):
    """Cohere Reranker Finetune Engine."""

    def __init__(
        self,
        cohere_api_key: str,
        train_file_name: str = "train.jsonl",
        val_file_name: Any = None,
        model_name: str = "exp_finetune",
        model_type: str = "RERANK",
        base_model: str = "english",
    ) -> None:
        """Init params."""
        # This will be None if 'cohere' module is not available
        cohere_spec = importlib.util.find_spec("cohere")

        if cohere_spec is not None:
            import cohere
        else:
            # Raise an ImportError if 'cohere' is not installed
            raise ImportError(
                "Cannot import cohere. Please install the package using `pip install cohere`."
            )

        self._model = cohere.Client(cohere_api_key)
        self._train_file_name = train_file_name
        self._val_file_name = val_file_name
        self._model_name = model_name
        self._model_type = model_type
        self._base_model = base_model
        self._finetune_model = None

    def finetune(self) -> None:
        """Finetune model."""
        from cohere.custom_model_dataset import JsonlDataset

        if self._val_file_name:
            # Uploading both train file and eval file
            dataset = JsonlDataset(
                train_file=self._train_file_name, eval_file=self._val_file_name
            )
        else:
            # Single Train File Upload:
            dataset = JsonlDataset(train_file=self._train_file_name)

        self._finetune_model = self._model.create_custom_model(
            name=self._model_name,
            dataset=dataset,
            model_type=self._model_type,
            base_model=self._base_model,
        )

    def get_finetuned_model(self) -> Any:
        """Gets finetuned model id."""
        if self._finetune_model is None:
            raise RuntimeError(
                "Finetuned model is not set yet. Please run the finetune method first."
            )
        return self._finetune_model
