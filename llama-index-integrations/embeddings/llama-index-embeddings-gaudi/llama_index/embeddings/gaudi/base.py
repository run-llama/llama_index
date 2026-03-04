# This file is adapted from
# https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/localGPT_inference/gaudi_utils/embeddings.py
#
import logging

from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.gaudi.utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
)

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)

DEFAULT_HUGGINGFACE_LENGTH = 512
DEFAULT_EMBED_INPUT_SIZE = -1
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "thenlper/gte-large"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GaudiSentenceTransformer(SentenceTransformer):
    """Child class that overrides the tokenize method from SentenceTransformer."""

    def __init__(self, model_name_or_path, embedding_input_size=-1, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)
        self.embedding_input_size = embedding_input_size

    def tokenize(self, texts):
        """Override tokenize method from SentenceTransformer."""
        return self._first_module().tokenizer(
            texts,
            max_length=self.max_seq_length
            if (
                self.embedding_input_size == -1
                or self.embedding_input_size > self.max_seq_length
            )
            else self.embedding_input_size,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )


class GaudiEmbedding(BaseEmbedding):
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

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        max_length: Optional[int] = DEFAULT_HUGGINGFACE_LENGTH,
        normalize: bool = True,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **model_kwargs,
    ) -> None:
        model = GaudiSentenceTransformer(
            model_name,
            cache_folder=get_cache_dir(),
            # prompts={
            #    "query": query_instruction
            #    or get_query_instruct_for_model_name(model_name),
            #    "text": text_instruction
            #    or get_text_instruct_for_model_name(model_name),
            # },
            **model_kwargs,
        )
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "GaudiEmbedding"

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
        return self._embed(query, prompt_name=None)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed(text, prompt_name=None)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, prompt_name=None)
