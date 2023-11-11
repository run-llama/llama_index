"""Cohere Reranker Finetuning Engine."""
import importlib.util
from typing import Any

from llama_index.finetuning.types import BaseCohereRerankerFinetuningEngine


class CohereRerankerFinetuneEngine(BaseCohereRerankerFinetuningEngine):
    """Cohere Reranker Finetune Engine."""

    def __init__(
        self,
        cohere_api_key: str,
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

        self.model = cohere.Client(cohere_api_key)

    def finetune(
        self,
        train_file_name: str = "train.jsonl",
        val_file_name: Any = None,
        model_name: str = "exp_finetune",
        model_type: str = "RERANK",
        base_model: str = "english",
    ) -> CustomModel:
        """Finetune model."""
        if val_file_name:
            # Uploading both train file and eval file
            dataset = JsonlDataset(train_file=train_file_name, eval_file=val_file_name)
        else:
            # Single Train File Upload:
            dataset = JsonlDataset(train_file=train_file_name)

        return self.model.create_custom_model(
            name=model_name,
            dataset=dataset,
            model_type=model_type,
            base_model=base_model,
        )

    def get_finetuned_model(self, finetune_model: CustomModel) -> str:
        """Gets finetuned model id."""
        return finetune_model.id
