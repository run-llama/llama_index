# This file is adapted from
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py

import logging
from typing import Any, List, Optional
from ipex_llm.transformers.convert import _optimize_pre, _optimize_post

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.ipex_llm.utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    BGE_MODELS,
    is_listed_model,
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)
from sentence_transformers import SentenceTransformer

DEFAULT_HUGGINGFACE_LENGTH = 512
logger = logging.getLogger(__name__)


class IpexLLMEmbedding(BaseEmbedding):
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for Hugging Face files."
    )

    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: str = "cpu",
        callback_manager: Optional[CallbackManager] = None,
        **model_kwargs,
    ):
        if device not in ["cpu", "xpu"] and not device.startswith("xpu:"):
            raise ValueError(
                "IpexLLMEmbedding currently only supports device to be 'cpu', 'xpu', "
                f"or 'xpu:<device_id>', but you have: {device}."
            )
        device = device

        cache_folder = cache_folder or get_cache_dir()

        if model_name is None:
            raise ValueError("The `model_name` argument must be provided.")
        if not is_listed_model(model_name, BGE_MODELS):
            bge_model_list_str = ", ".join(BGE_MODELS)
            logger.warning(
                "IpexLLMEmbedding currently only provides optimization for "
                f"Hugging Face BGE models, which are: {bge_model_list_str}"
            )

        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            prompts={
                "query": query_instruction
                or get_query_instruct_for_model_name(model_name),
                "text": text_instruction
                or get_text_instruct_for_model_name(model_name),
            },
            **model_kwargs,
        )

        # Apply ipex-llm optimizations
        model = _optimize_pre(self._model)
        model = _optimize_post(self._model)
        if device == "xpu":
            # TODO: apply `ipex_llm.optimize_model`
            model = model.half().to(device)

        if max_length:
            model.max_seq_length = max_length
        else:
            max_length = model.max_seq_length

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )
        self._model = model
        self._device = device

    @classmethod
    def class_name(cls) -> str:
        return "IpexLLMEmbedding"

    def _embed(
        self,
        sentences: List[str],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed sentences."""
        return self._model.encode(
            sentences,
            batch_size=self.embed_batch_size,
            prompt_name=prompt_name,
            normalize_embeddings=self.normalize,
        ).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed(query, prompt_name="query")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed(text, prompt_name="text")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, prompt_name="text")
