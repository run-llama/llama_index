from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding


class OptimumEmbedding(BaseEmbedding):
    folder_name: str = Field(description="Folder name to load from.")
    max_length: int = Field(description="Maximum length of input.")
    pooling: str = Field(description="Pooling strategy. One of ['cls', 'mean'].")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files."
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        folder_name: str,
        pooling: str = "cls",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "OptimumEmbedding requires transformers to be installed.\n"
                "Please install transformers with "
                "`pip install transformers optimum[exporters]`."
            )

        self._model = model or ORTModelForFeatureExtraction.from_pretrained(folder_name)
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(folder_name)

        if max_length is None:
            try:
                max_length = int(self._model.config.max_position_embeddings)
            except Exception:
                raise ValueError(
                    "Unable to find max_length from model config. "
                    "Please provide max_length."
                )

        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling {pooling} not supported.")

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            folder_name=folder_name,
            max_length=max_length,
            pooling=pooling,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OptimumEmbedding"

    @classmethod
    def create_and_save_optimum_model(
        cls,
        model_name_or_path: str,
        output_path: str,
        export_kwargs: Optional[dict] = None,
    ) -> None:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "OptimumEmbedding requires transformers to be installed.\n"
                "Please install transformers with "
                "`pip install transformers optimum[exporters]`."
            )

        export_kwargs = export_kwargs or {}
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name_or_path, export=True, **export_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(
            f"Saved optimum model to {output_path}. Use it with "
            f"`embed_model = OptimumEmbedding(folder_name='{output_path}')`."
        )

    def _format_query_text(self, query_text: str) -> str:
        """Format query text."""
        return f"{self.query_instruction} {query_text}".strip()

    def _format_text(self, text: str) -> str:
        """Format text."""
        return f"{self.text_instruction} {text}".strip()

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        import torch

        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _cls_pooling(self, model_output: list) -> Any:
        """Use the CLS token as the pooling token."""
        return model_output[0][:, 0]

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        encoded_input = self._tokenizer(
            sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_output = self._model(**encoded_input)

        if self.pooling == "cls":
            return self._cls_pooling(model_output).tolist()
        else:
            return self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = self._format_query_text(query)
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = self._format_text(text)
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [self._format_text(text) for text in texts]
        return self._embed(texts)
