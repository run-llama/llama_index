"""Cohere Reranker Finetuning Engine."""
import importlib.util
import os
from typing import Optional

from llama_index.finetuning.types import BaseCohereRerankerFinetuningEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank


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
            import cohere
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

    def get_finetuned_model(self, top_n: int = 5) -> CohereRerank:
        """Gets finetuned model id."""
        if self._finetune_model is None:
            raise RuntimeError(
                "Finetuned model is not set yet. Please run the finetune method first."
            )
        return CohereRerank(
            model=self._finetune_model.id, top_n=top_n, api_key=self.api_key
        )
