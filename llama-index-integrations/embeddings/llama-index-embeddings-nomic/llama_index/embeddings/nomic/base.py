from enum import Enum
from typing import Any, List, Optional, Union

from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.embeddings.huggingface.pooling import Pooling
import torch
import logging

DEFAULT_HUGGINGFACE_LENGTH = 512
logger = logging.getLogger(__name__)


class NomicAITaskType(str, Enum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


TASK_TYPES = [
    NomicAITaskType.SEARCH_QUERY,
    NomicAITaskType.SEARCH_DOCUMENT,
    NomicAITaskType.CLUSTERING,
    NomicAITaskType.CLASSIFICATION,
]


class NomicEmbedding(BaseEmbedding):
    """NomicEmbedding uses the Nomic API to generate embeddings."""

    # Instance variables initialized via Pydantic's mechanism
    query_task_type: Optional[str] = Field(description="Query Embedding prefix")
    document_task_type: Optional[str] = Field(description="Document Embedding prefix")
    dimensionality: Optional[int] = Field(description="Dimension of the Embedding")
    model_name: str = Field(description="Embedding model name")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1",
        embed_batch_size: int = 32,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        query_task_type: Optional[str] = "search_query",
        document_task_type: Optional[str] = "search_document",
        dimensionality: Optional[int] = 768,
        **kwargs: Any,
    ) -> None:
        if query_task_type not in TASK_TYPES or document_task_type not in TASK_TYPES:
            raise ValueError(
                f"Invalid task type {query_task_type}, {document_task_type}. Must be one of {TASK_TYPES}"
            )

        try:
            import nomic
            from nomic import embed
        except ImportError:
            raise ImportError(
                "NomicEmbedding requires the 'nomic' package to be installed.\n"
                "Please install it with `pip install nomic`."
            )

        if api_key is not None:
            nomic.cli.login(api_key)
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            _model=embed,
            query_task_type=query_task_type,
            document_task_type=document_task_type,
            dimensionality=dimensionality,
            **kwargs,
        )
        self._model = embed
        self.model_name = model_name
        self.query_task_type = query_task_type
        self.document_task_type = document_task_type
        self.dimensionality = dimensionality

    @classmethod
    def class_name(cls) -> str:
        return "NomicEmbedding"

    def _embed(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Embed sentences using NomicAI."""
        result = self._model.text(
            texts,
            model=self.model_name,
            task_type=task_type,
            dimensionality=self.dimensionality,
        )
        return result["embeddings"]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query], task_type=self.query_task_type)[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text], task_type=self.document_task_type)[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, task_type=self.document_task_type)


class NomicHFEmbedding(HuggingFaceEmbedding):
    tokenizer_name: str = Field(description="Tokenizer name from HuggingFace.")
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    pooling: Pooling = Field(default=Pooling.MEAN, description="Pooling strategy.")
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files."
    )
    dimensionality: Optional[int] = Field(description="Dimensionality of embedding")

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
        dimensionality: int = 768,
    ):
        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            pooling=pooling,
            max_length=max_length,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            normalize=normalize,
            model=model,
            tokenizer=tokenizer,
            embed_batch_size=embed_batch_size,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            device=device,
            callback_manager=callback_manager,
        )
        self.dimensionality = dimensionality
        self._model.eval()

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        encoded_input = self._tokenizer(
            sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # pop token_type_ids
        encoded_input.pop("token_type_ids", None)

        # move tokenizer inputs to device
        encoded_input = {
            key: val.to(self._device) for key, val in encoded_input.items()
        }

        with torch.no_grad():
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
            import torch.nn.functional as F

            if self.model_name == "nomic-ai/nomic-embed-text-v1.5":
                emb_ln = F.layer_norm(
                    embeddings, normalized_shape=(embeddings.shape[1],)
                )
                embeddings = emb_ln[:, : self.dimensionality]

            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()
