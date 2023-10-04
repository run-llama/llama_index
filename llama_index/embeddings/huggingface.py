from typing import Any, List, Optional, Literal

from llama_index.bridge.pydantic import PrivateAttr, Field
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
    pooling: Literal["mean", "cls", "weighted_mean"] = Field(
        description="Pooling strategy. One of ['mean', 'cls', 'weighted_mean']."
    )
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_dir: Optional[str] = Field(description="Cache folder for huggingface files.")

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        pooling: str = "mean",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        model_args: dict = {},
        tokenizer_args: dict = {},
    ):
        try:
            from transformers import AutoTokenizer, AutoConfig
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
            config = AutoConfig.from_pretrained(
                model_name_or_path or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
                cache_dir=cache_dir,
                **model_args,
            )
            model_name_or_path = (
                model_name_or_path or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            )
            self._model = self._load_model(
                model_name_or_path,
                config,
                cache_dir,
                **model_args,
            ).to(device)
        else:
            self._model = model

        if tokenizer is None:
            tokenizer_name_or_path = (
                model_name_or_path
                or tokenizer_name_or_path
                or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, cache_dir=cache_dir, **tokenizer_args
            )
        else:
            self._tokenizer = tokenizer

        if max_length is None:
            max_position_embeddings = self._model.config.max_position_embeddings or 0
            tokenizer_max_length = self._tokenizer.model_max_length or 0
            max_length = min(max_position_embeddings, tokenizer_max_length)
            if max_length == 0:
                raise ValueError(
                    "Unable to infer `max_position_embeddings` from the model config or `model_max_length` from the tokenizer config. "
                    "Please provide `max_length` to initialise the `HuggingFaceEmbedding` class."
                )

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name_or_path,
            tokenizer_name=tokenizer_name_or_path,
            max_length=max_length,
            pooling=pooling,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    def _load_model(
        self,
        model_name_or_path: str,
        config: Any,
        cache_dir: Optional[str] = None,
        **model_args,
    ):
        """Loads the transformer model"""
        from transformers import AutoModel, T5Config, MT5Config

        if isinstance(config, T5Config):
            self._model = self._load_t5_model(
                model_name_or_path, config, cache_dir, **model_args
            )
        elif isinstance(config, MT5Config):
            self._model = self._load_mt5_model(
                model_name_or_path, config, cache_dir, **model_args
            )
        else:
            self._model = AutoModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )

    def _load_t5_model(
        self,
        model_name_or_path: str,
        config: Any,
        cache_dir: Optional[str] = None,
        **model_args,
    ):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        return T5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def _load_mt5_model(
        self,
        model_name_or_path: str,
        config: Any,
        cache_dir: Optional[str] = None,
        **model_args,
    ):
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel

        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        return MT5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

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

    def _weighted_mean_pooling(
        self, model_output: Any, attention_mask: Any, device: str
    ) -> Any:
        """Weighted mean pooling assigns higher weighted to tokens attend later, such as SGPT."""
        import torch

        token_embeddings = model_output[0]
        weights = (
            torch.arange(start=1, end=token_embeddings.shape[1] + 1, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded * weights, 1
        ) / torch.clamp(torch.sum(input_mask_expanded * weights, dim=1), min=1e-9)

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
        elif self.pooling == "weighted_mean":
            return self._weighted_mean_pooling(
                model_output, encoded_input["attention_mask"], self._device
            )
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
