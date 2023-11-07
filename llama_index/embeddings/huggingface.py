import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.embeddings.huggingface_utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    format_query,
    format_text,
)
from llama_index.embeddings.pooling import Pooling
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.utils import get_cache_dir, infer_torch_device

if TYPE_CHECKING:
    import torch


class HuggingFaceEmbedding(BaseEmbedding):
    tokenizer_name: str = Field(description="Tokenizer name from HuggingFace.")
    max_length: int = Field(description="Maximum length of input.")
    pooling: Pooling = Field(default=Pooling.CLS, description="Pooling strategy.")
    normalize: str = Field(default=True, description="Normalize embeddings or not.")
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
        pooling: Union[str, Pooling] = "cls",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
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

        self._device = device or infer_torch_device()

        cache_folder = cache_folder or get_cache_dir()

        if model is None:  # Use model_name with AutoModel
            model_name = (
                model_name
                if model_name is not None
                else DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            )
            model = AutoModel.from_pretrained(
                model_name, cache_dir=cache_folder, trust_remote_code=trust_remote_code
            )
        elif model_name is None:  # Extract model_name from model
            model_name = model.name_or_path
        self._model = model.to(self._device)

        if tokenizer is None:  # Use tokenizer_name with AutoTokenizer
            tokenizer_name = (
                model_name or tokenizer_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, cache_dir=cache_folder
            )
        elif tokenizer_name is None:  # Extract tokenizer_name from model
            tokenizer_name = tokenizer.name_or_path
        self._tokenizer = tokenizer

        if max_length is None:
            try:
                max_length = int(self._model.config.max_position_embeddings)
            except AttributeError as exc:
                raise ValueError(
                    "Unable to find max_length from model config. Please specify max_length."
                ) from exc

        if isinstance(pooling, str):
            try:
                pooling = Pooling(pooling)
            except ValueError as exc:
                raise NotImplementedError(
                    f"Pooling {pooling} unsupported, please pick one in"
                    f" {[p.value for p in Pooling]}."
                ) from exc

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    def _mean_pooling(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """Mean Pooling - Take attention mask into account for correct averaging."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        numerator = (token_embeddings * input_mask_expanded).sum(1)
        return numerator / input_mask_expanded.sum(1).clamp(min=1e-9)

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

        if self.pooling == Pooling.CLS:
            context_layer: "torch.Tensor" = model_output[0]
            embeddings = self.pooling.cls_pooling(context_layer)
        else:
            embeddings = self._mean_pooling(
                token_embeddings=model_output[0],
                attention_mask=encoded_input["attention_mask"],
            )

        if self.normalize:
            import torch

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = format_query(query, self.model_name, self.query_instruction)
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = format_text(text, self.model_name, self.text_instruction)
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return self._embed(texts)


class HuggingFaceInferenceAPIEmbedding(HuggingFaceInferenceAPI, BaseEmbedding):  # type: ignore[misc]
    """
    Wrapper on the Hugging Face's Inference API for embeddings.

    Overview of the design:
    - Uses the feature extraction task: https://huggingface.co/tasks/feature-extraction
    """

    pooling: Optional[Pooling] = Field(
        default=Pooling.CLS,
        description=(
            "Optional pooling technique to use with embeddings capability, if"
            " the model's raw output needs pooling."
        ),
    )
    query_instruction: Optional[str] = Field(
        default=None,
        description=(
            "Instruction to prepend during query embedding."
            " Use of None means infer the instruction based on the model."
            " Use of empty string will defeat instruction prepending entirely."
        ),
    )
    text_instruction: Optional[str] = Field(
        default=None,
        description=(
            "Instruction to prepend during text embedding."
            " Use of None means infer the instruction based on the model."
            " Use of empty string will defeat instruction prepending entirely."
        ),
    )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceInferenceAPIEmbedding"

    async def _async_embed_single(self, text: str) -> Embedding:
        embedding = (await self._async_client.feature_extraction(text)).squeeze(axis=0)
        if len(embedding.shape) == 1:  # Some models pool internally
            return list(embedding)
        try:
            return list(self.pooling(embedding))  # type: ignore[misc]
        except TypeError as exc:
            raise ValueError(
                f"Pooling is required for {self.model_name} because it returned"
                " a > 1-D value, please specify pooling as not None."
            ) from exc

    async def _async_embed_bulk(self, texts: Sequence[str]) -> List[Embedding]:
        """
        Embed a sequence of text, in parallel and asynchronously.

        NOTE: this uses an externally created asyncio event loop.
        """
        tasks = [self._async_embed_single(text) for text in texts]
        return await asyncio.gather(*tasks)

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        NOTE: a new asyncio event loop is created internally for this.
        """
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the text query synchronously.

        NOTE: a new asyncio event loop is created internally for this.
        """
        return asyncio.run(self._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously and in parallel.

        NOTE: a new asyncio event loop is created internally for this.
        """
        loop = asyncio.new_event_loop()
        try:
            tasks = [
                loop.create_task(self._aget_text_embedding(text)) for text in texts
            ]
            loop.run_until_complete(asyncio.wait(tasks))
        finally:
            loop.close()
        return [task.result() for task in tasks]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._async_embed_single(
            text=format_query(query, self.model_name, self.query_instruction)
        )

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._async_embed_single(
            text=format_text(text, self.model_name, self.text_instruction)
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return await self._async_embed_bulk(
            texts=[
                format_text(text, self.model_name, self.text_instruction)
                for text in texts
            ]
        )


HuggingFaceInferenceAPIEmbeddings = HuggingFaceInferenceAPIEmbedding
