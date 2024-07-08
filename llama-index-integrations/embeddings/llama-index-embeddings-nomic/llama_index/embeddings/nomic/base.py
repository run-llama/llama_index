from enum import Enum
from typing import Any, List, Optional, Union

import nomic
import nomic.embed
import warnings
from PIL import Image
import torch
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.pooling import Pooling
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType

DEFAULT_HUGGINGFACE_LENGTH = 512


class NomicTaskType(str, Enum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


class NomicInferenceMode(str, Enum):
    REMOTE = "remote"
    LOCAL = "local"
    DYNAMIC = "dynamic"


class NomicEmbedding(MultiModalEmbedding):
    """NomicEmbedding uses the Nomic API to generate embeddings."""

    query_task_type: Optional[NomicTaskType] = Field(
        description="Task type for queries",
    )
    document_task_type: Optional[NomicTaskType] = Field(
        description="Task type for documents",
    )
    dimensionality: Optional[int] = Field(
        description="Embedding dimension, for use with Matryoshka-capable models",
    )
    model_name: str = Field(description="Embedding model name")
    vision_model_name: Optional[str] = Field(
        description="Vision model name for multimodal embeddings",
    )
    inference_mode: NomicInferenceMode = Field(
        description="Whether to generate embeddings locally",
    )
    device: Optional[str] = Field(description="Device to use for local embeddings")

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1",
        vision_model_name: Optional[str] = "nomic-embed-vision-v1",
        embed_batch_size: int = 32,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        query_task_type: Optional[str] = "search_query",
        document_task_type: Optional[str] = "search_document",
        dimensionality: Optional[int] = 768,
        inference_mode: str = "remote",
        device: Optional[str] = None,
    ):
        if api_key is not None:
            nomic.login(api_key)

        super().__init__(
            model_name=model_name,
            vision_model_name=vision_model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            query_task_type=query_task_type,
            document_task_type=document_task_type,
            dimensionality=dimensionality,
            inference_mode=inference_mode,
            device=device,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NomicEmbedding"

    def load_images(self, image_paths: List[ImageType]) -> List[Image.Image]:
        """Load images from the specified paths."""
        return [Image.open(image_path).convert("RGB") for image_path in image_paths]

    def _embed_text(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        result = nomic.embed.text(
            texts,
            model=self.model_name,
            task_type=task_type,
            dimensionality=self.dimensionality,
            inference_mode=self.inference_mode,
            device=self.device,
        )
        return result["embeddings"]

    def _embed_image(self, images_paths: List[ImageType]) -> List[List[float]]:
        images = self.load_images(images_paths)
        result = nomic.embed.image(images, model=self.vision_model_name)
        return result["embeddings"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_text([query], task_type=self.query_task_type)[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        self._warn_async()
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_text([text], task_type=self.document_task_type)[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        self._warn_async()
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_text(texts, task_type=self.document_task_type)

    def _get_image_embedding(self, image: ImageType) -> List[float]:
        return self._embed_image([image])[0]

    async def _aget_image_embedding(self, image: ImageType) -> List[float]:
        self._warn_async()
        return self._get_image_embedding(image)

    def _get_image_embeddings(self, images: List[ImageType]) -> List[List[float]]:
        return self._embed_image(images)

    def _warn_async(self) -> None:
        warnings.warn(
            f"{self.class_name()} does not implement async embeddings, falling back to sync method.",
        )


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
