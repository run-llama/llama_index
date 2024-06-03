from enum import Enum
from typing import List, Optional

import nomic
import nomic.embed
import warnings
from PIL import Image
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType


class NomicTaskType(str, Enum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


class NomicInferenceMode(str, Enum):
    REMOTE = "remote"
    LOCAL = "local"
    DYNAMIC = "dynamic"


class NomicMultiModalEmbedding(MultiModalEmbedding):
    """NomicMultiModalEmbedding uses Nomic API for encoding text and image for Multi-Modal purpose."""

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
    inference_mode: NomicInferenceMode = Field(
        description="Whether to generate embeddings locally",
    )
    device: Optional[str] = Field(description="Device to use for local embeddings")

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1",
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
        return "NomicMultiModalEmbedding"

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
        result = nomic.embed.image(images)
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

    def _get_imag_embeddings(self, images: List[ImageType]) -> List[List[float]]:
        return self._embed_image(images)

    def _warn_async() -> None:
        warnings.warn(
            f"{self.class_name()} does not implement async embeddings, falling back to sync method.",
        )
