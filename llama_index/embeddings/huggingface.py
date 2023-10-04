from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.embeddings.huggingface_utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)


class HuggingFaceEmbedding(BaseEmbedding):
    tokenizer_name: str = Field(description="Tokenizer name from HuggingFace.")
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
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        pooling: str = "cls",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFaceEmbedding requires transformers to be installed.\n"
                "Please install transformers with `pip install transformers`."
            )

        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if model is None:
            model_name = model_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            self._model = AutoModel.from_pretrained(
                model_name, cache_dir=cache_folder
            ).to(device)
        else:
            self._model = model

        if tokenizer is None:
            tokenizer_name = (
                model_name or tokenizer_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, cache_dir=cache_folder
            )
        else:
            self._tokenizer = tokenizer

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
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            pooling=pooling,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    def _format_query_text(self, query_text: str) -> str:
        """Format query text."""
        instruction = self.text_instruction

        if instruction is None:
            instruction = get_query_instruct_for_model_name(self.model_name)

        return f"{instruction} {query_text}".strip()

    def _format_text(self, text: str) -> str:
        """Format text."""
        instruction = self.text_instruction

        if instruction is None:
            instruction = get_text_instruct_for_model_name(self.model_name)

        return f"{instruction} {text}".strip()

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

        # move tokenizer inputs to device
        encoded_input = {
            key: val.to(self._device) for key, val in encoded_input.items()
        }

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
