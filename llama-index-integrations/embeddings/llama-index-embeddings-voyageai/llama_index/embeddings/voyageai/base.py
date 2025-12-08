"""Voyage embeddings file."""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import voyageai
from PIL import Image

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.schema import ImageType

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1000

MULTIMODAL_MODELS = ["voyage-multimodal-3"]
CONTEXT_MODELS = ["voyage-context-3"]

SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "webp", "gif"}

VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-context-3": 32_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,  # voyage-3.5 supports up to 320k tokens per batch
    "voyage-2": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
}


class VoyageEmbedding(MultiModalEmbedding):
    """
    Class for Voyage embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "voyage-01".

        voyage_api_key (Optional[str]): Voyage API key. Defaults to None.
            You can either specify the key here or store it as an environment variable.

    """

    _client: voyageai.Client = PrivateAttr(None)
    _aclient: voyageai.AsyncClient = PrivateAttr()
    truncation: Optional[bool] = None
    output_dtype: Optional[str] = None
    output_dimension: Optional[int] = None

    def __init__(
        self,
        model_name: str,
        voyage_api_key: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        truncation: Optional[bool] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        if model_name in [
            "voyage-01",
            "voyage-lite-01",
            "voyage-lite-01-instruct",
            "voyage-02",
            "voyage-2",
            "voyage-lite-02-instruct",
            "voyage-large-2",
            "voyage-large-2-instruct",
            "voyage-multilingual-2",
            "voyage-3",
            "voyage-3-lite",
        ]:
            logger.warning(
                f"{model_name} is not the latest model by Voyage AI. Please note that `model_name` "
                "will be a required argument in the future. We recommend setting it explicitly. Please see "
                "https://docs.voyageai.com/docs/embeddings for the latest models offered by Voyage AI."
            )

        if embed_batch_size is None:
            embed_batch_size = MAX_BATCH_SIZE

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = voyageai.Client(api_key=voyage_api_key)
        self._aclient = voyageai.AsyncClient(api_key=voyage_api_key)
        self.truncation = truncation
        self.output_dtype = output_dtype
        self.output_dimension = output_dimension

    @classmethod
    def class_name(cls) -> str:
        return "VoyageEmbedding"

    @staticmethod
    def _validate_image_format(file_type: str) -> bool:
        """Validate image format."""
        return file_type.lower() in SUPPORTED_IMAGE_FORMATS

    @classmethod
    def _texts_to_content(cls, input_strs: List[str]) -> List[dict]:
        return [{"content": [{"type": "text", "text": x}]} for x in input_strs]

    def _build_batches(
        self, texts: List[str]
    ) -> Generator[Tuple[List[str], int], None, None]:
        """Generate batches of texts based on token limits."""
        max_tokens_per_batch = VOYAGE_TOTAL_TOKEN_LIMITS.get(self.model_name, 120_000)
        index = 0

        while index < len(texts):
            batch: List[str] = []
            batch_tokens = 0
            while (
                index < len(texts)
                and len(batch) < min(self.embed_batch_size, MAX_BATCH_SIZE)
                and batch_tokens < max_tokens_per_batch
            ):
                n_tokens = len(
                    self._client.tokenize([texts[index]], model=self.model_name)[0]
                )
                if batch_tokens + n_tokens > max_tokens_per_batch and len(batch) > 0:
                    break
                batch_tokens += n_tokens
                batch.append(texts[index])
                index += 1

            yield batch, len(batch)

    def _image_to_content(self, image_input: Union[str, Path, BytesIO]) -> Image:
        """Convert an image to a base64 Data URL."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(str(image_input))
            # If it's a string or Path, assume it's a file path
            image_path = str(image_input)
            file_extension = os.path.splitext(image_path)[1][1:].lower()
        elif isinstance(image_input, BytesIO):
            # If it's a BytesIO, use it directly
            image = Image.open(image_input)
            file_extension = image.format.lower()
            image_input.seek(0)  # Reset the BytesIO stream to the beginning
        else:
            raise ValueError("Unsupported input type. Must be a file path or BytesIO.")

        if self._validate_image_format(file_extension):
            return image
        else:
            raise ValueError(f"Unsupported image format: {file_extension}")

    def _embed_image(
        self, image_path: ImageType, input_type: Optional[str] = None
    ) -> List[float]:
        """Embed images using VoyageAI."""
        if self.model_name not in MULTIMODAL_MODELS:
            raise ValueError(
                f"{self.model_name} is not a valid multi-modal embedding model. Supported models are {MULTIMODAL_MODELS}"
            )
        processed_image = self._image_to_content(image_path)
        return self._client.multimodal_embed(
            model=self.model_name,
            inputs=[[processed_image]],
            input_type=input_type,
            truncation=self.truncation,
        ).embeddings[0]

    async def _aembed_image(
        self, image_path: ImageType, input_type: Optional[str] = None
    ) -> List[float]:
        """Embed images using VoyageAI."""
        if self.model_name not in MULTIMODAL_MODELS:
            raise ValueError(
                f"{self.model_name} is not a valid multi-modal embedding model. Supported models are {MULTIMODAL_MODELS}"
            )
        processed_image = self._image_to_content(image_path)
        return (
            await self._aclient.multimodal_embed(
                model=self.model_name,
                inputs=[[processed_image]],
                input_type=input_type,
                truncation=self.truncation,
            )
        ).embeddings[0]

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._embed_image(img_file_path)

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return await self._aembed_image(img_file_path)

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed texts with dynamic batching based on token limits."""
        embeddings: List[List[float]] = []

        for batch, _ in self._build_batches(texts):
            if self.model_name in CONTEXT_MODELS:
                r = self._client.contextualized_embed(
                    inputs=[batch],
                    model=self.model_name,
                    input_type=input_type,
                    output_dtype=self.output_dtype,
                    output_dimension=self.output_dimension,
                ).results
                embeddings.extend(r[0].embeddings)
            elif self.model_name in MULTIMODAL_MODELS:
                batch_embeddings = self._client.multimodal_embed(
                    inputs=self._texts_to_content(batch),
                    model=self.model_name,
                    input_type=input_type,
                    truncation=self.truncation,
                ).embeddings
                embeddings.extend(batch_embeddings)
            else:
                batch_embeddings = self._client.embed(
                    batch,
                    model=self.model_name,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dtype=self.output_dtype,
                    output_dimension=self.output_dimension,
                ).embeddings
                embeddings.extend(batch_embeddings)

        return embeddings

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Asynchronously embed texts with dynamic batching based on token limits."""
        embeddings: List[List[float]] = []

        for batch, _ in self._build_batches(texts):
            if self.model_name in CONTEXT_MODELS:
                ar = await self._aclient.contextualized_embed(
                    inputs=[batch],
                    model=self.model_name,
                    input_type=input_type,
                    output_dtype=self.output_dtype,
                    output_dimension=self.output_dimension,
                )
                r = ar.results
                embeddings.extend(r[0].embeddings)
            elif self.model_name in MULTIMODAL_MODELS:
                r = await self._aclient.multimodal_embed(
                    inputs=self._texts_to_content(batch),
                    model=self.model_name,
                    input_type=input_type,
                    truncation=self.truncation,
                )
                embeddings.extend(r.embeddings)
            else:
                r = await self._aclient.embed(
                    batch,
                    model=self.model_name,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dtype=self.output_dtype,
                    output_dimension=self.output_dimension,
                )
                embeddings.extend(r.embeddings)

        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query], input_type="query")[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        r = await self._aembed([query], input_type="query")
        return r[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text], input_type="document")[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        r = await self._aembed([text], input_type="document")
        return r[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, input_type="document")

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._aembed(texts, input_type="document")

    def get_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Get general text embedding with input_type."""
        return self._embed([text], input_type=input_type)[0]

    async def aget_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Asynchronously get general text embedding with input_type."""
        r = await self._aembed([text], input_type=input_type)
        return r[0]
