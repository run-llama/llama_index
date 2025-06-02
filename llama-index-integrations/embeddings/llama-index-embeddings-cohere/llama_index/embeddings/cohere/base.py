from enum import Enum
from typing import Any, List, Optional, Union

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, Embedding
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
import cohere
import httpx
import base64
from io import BytesIO
from pathlib import Path
from llama_index.core.schema import ImageType
from PIL import Image


# Enums for validation and type safety
class CohereAIModelName(str, Enum):
    ENGLISH_V3 = "embed-english-v3.0"
    ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    ENGLISH_V2 = "embed-english-v2.0"
    ENGLISH_LIGHT_V2 = "embed-english-light-v2.0"
    MULTILINGUAL_V2 = "embed-multilingual-v2.0"
    EMBED_V4 = "embed-v4.0"


class CohereAIInputType(str, Enum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class CohereAITruncate(str, Enum):
    START = "START"
    END = "END"
    NONE = "NONE"


# convenient shorthand
CAMN = CohereAIModelName
CAIT = CohereAIInputType
CAT = CohereAITruncate

# This list would be used for model name and input type validation
VALID_MODEL_INPUT_TYPES = {
    CAMN.EMBED_V4: [
        None,
        CAIT.SEARCH_QUERY,
        CAIT.SEARCH_DOCUMENT,
        CAIT.CLASSIFICATION,
        CAIT.CLUSTERING,
    ],
    CAMN.ENGLISH_V3: [
        None,
        CAIT.SEARCH_QUERY,
        CAIT.SEARCH_DOCUMENT,
        CAIT.CLASSIFICATION,
        CAIT.CLUSTERING,
    ],
    CAMN.ENGLISH_LIGHT_V3: [
        None,
        CAIT.SEARCH_QUERY,
        CAIT.SEARCH_DOCUMENT,
        CAIT.CLASSIFICATION,
        CAIT.CLUSTERING,
    ],
    CAMN.MULTILINGUAL_V3: [
        None,
        CAIT.SEARCH_QUERY,
        CAIT.SEARCH_DOCUMENT,
        CAIT.CLASSIFICATION,
        CAIT.CLUSTERING,
    ],
    CAMN.MULTILINGUAL_LIGHT_V3: [
        None,
        CAIT.SEARCH_QUERY,
        CAIT.SEARCH_DOCUMENT,
        CAIT.CLASSIFICATION,
        CAIT.CLUSTERING,
    ],
    CAMN.ENGLISH_V2: [None],
    CAMN.ENGLISH_LIGHT_V2: [None],
    CAMN.MULTILINGUAL_V2: [None],
}

# v4 models support input type parameter
V4_MODELS = [CAMN.EMBED_V4]

# v3 models require an input_type field
# supported models for multimodal embeddings
V3_MODELS = [
    CAMN.ENGLISH_V3,
    CAMN.ENGLISH_LIGHT_V3,
    CAMN.MULTILINGUAL_V3,
    CAMN.MULTILINGUAL_LIGHT_V3,
]

# This list would be used for model name and embedding types validation
# Embedding type can be float/ int8/ uint8/ binary/ ubinary based on model.
VALID_MODEL_EMBEDDING_TYPES = {
    CAMN.EMBED_V4: ["float", "int8", "uint8", "binary", "ubinary"],
    CAMN.ENGLISH_V3: ["float", "int8", "uint8", "binary", "ubinary"],
    CAMN.ENGLISH_LIGHT_V3: ["float", "int8", "uint8", "binary", "ubinary"],
    CAMN.MULTILINGUAL_V3: ["float", "int8", "uint8", "binary", "ubinary"],
    CAMN.MULTILINGUAL_LIGHT_V3: ["float", "int8", "uint8", "binary", "ubinary"],
    CAMN.ENGLISH_V2: ["float"],
    CAMN.ENGLISH_LIGHT_V2: ["float"],
    CAMN.MULTILINGUAL_V2: ["float"],
}

VALID_TRUNCATE_OPTIONS = [CAT.START, CAT.END, CAT.NONE]

# supported image formats
SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "webp", "gif"}

# Maximum batch size for Cohere API
MAX_EMBED_BATCH_SIZE = 96


# Assuming BaseEmbedding is a Pydantic model and handles its own initializations
class CohereEmbedding(MultiModalEmbedding):
    """CohereEmbedding uses the Cohere API to generate embeddings for text."""

    # Instance variables initialized via Pydantic's mechanism
    api_key: str = Field(description="The Cohere API key.")
    base_url: Optional[str] = Field(
        default=None, description="The endpoint to use. Defaults to the Cohere API."
    )
    truncate: str = Field(description="Truncation type - START/ END/ NONE")
    input_type: Optional[str] = Field(
        default=None,
        description="Model Input type. If not provided, search_document and search_query are used when needed.",
    )
    embedding_type: str = Field(
        description="Embedding type. If not provided float embedding_type is used when needed."
    )

    _client: cohere.Client = PrivateAttr()
    _async_client: cohere.AsyncClient = PrivateAttr()
    _timeout: Optional[float] = PrivateAttr()
    _httpx_client: Optional[httpx.Client] = PrivateAttr()
    _httpx_async_client: Optional[httpx.AsyncClient] = PrivateAttr()

    def __init__(
        self,
        # deprecated
        cohere_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "embed-english-v3.0",
        truncate: str = "END",
        input_type: Optional[str] = None,
        embedding_type: str = "float",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        httpx_async_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        A class representation for generating embeddings using the Cohere API.

        Args:
            truncate (str): A string indicating the truncation strategy to be applied to input text. Possible values
                        are 'START', 'END', or 'NONE'.
            input_type (Optional[str]): An optional string that specifies the type of input provided to the model.
                                    This is model-dependent and could be one of the following: 'search_query',
                                    'search_document', 'classification', or 'clustering'.
            model_name (str): The name of the model to be used for generating embeddings. The class ensures that
                          this model is supported and that the input type provided is compatible with the model.
            embed_batch_size (int): The batch size for embedding generation. Maximum allowed value is 96 (MAX_EMBED_BATCH_SIZE)
                                   due to Cohere API limitations. Defaults to DEFAULT_EMBED_BATCH_SIZE.

        """
        # Validate model_name and input_type
        if model_name not in VALID_MODEL_INPUT_TYPES:
            raise ValueError(f"{model_name} is not a valid model name")

        if input_type not in VALID_MODEL_INPUT_TYPES[model_name]:
            raise ValueError(
                f"{input_type} is not a valid input type for the provided model."
            )
        if embedding_type not in VALID_MODEL_EMBEDDING_TYPES[model_name]:
            raise ValueError(
                f"{embedding_type} is not a embedding type for the provided model."
            )

        if truncate not in VALID_TRUNCATE_OPTIONS:
            raise ValueError(f"truncate must be one of {VALID_TRUNCATE_OPTIONS}")

        # Validate embed_batch_size
        if embed_batch_size > MAX_EMBED_BATCH_SIZE:
            raise ValueError(
                f"embed_batch_size {embed_batch_size} exceeds the maximum allowed value of {MAX_EMBED_BATCH_SIZE} for Cohere API"
            )

        super().__init__(
            api_key=api_key or cohere_api_key,
            model_name=model_name,
            input_type=input_type,
            embedding_type=embedding_type,
            truncate=truncate,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            num_workers=num_workers,
            **kwargs,
        )

        self.base_url = base_url

        self._client = None
        self._async_client = None
        self._timeout = timeout
        self._httpx_client = httpx_client
        self._httpx_async_client = httpx_async_client

    def _get_client(self) -> cohere.ClientV2:
        if self._client is None:
            self._client = cohere.ClientV2(
                api_key=self.api_key,
                client_name="llama_index",
                base_url=self.base_url,
                timeout=self._timeout,
                httpx_client=self._httpx_client,
            )

        return self._client

    def _get_async_client(self) -> cohere.AsyncClientV2:
        if self._async_client is None:
            self._async_client = cohere.AsyncClientV2(
                api_key=self.api_key,
                client_name="llama_index",
                base_url=self.base_url,
                timeout=self._timeout,
                httpx_client=self._httpx_async_client,
            )

        return self._async_client

    @classmethod
    def class_name(cls) -> str:
        return "CohereEmbedding"

    def _image_to_base64_data_url(self, image_input: Union[str, Path, BytesIO]) -> str:
        """Convert an image to a base64 Data URL."""
        if isinstance(image_input, (str, Path)):
            # If it's a string or Path, assume it's a file path
            image_path = Path(image_input)
            file_extension = image_path.suffix.lower().replace(".", "")
            with open(image_path, "rb") as f:
                image_data = f.read()
        elif isinstance(image_input, BytesIO):
            # If it's a BytesIO, use it directly
            image = Image.open(image_input)
            file_extension = image.format.lower()
            image_input.seek(0)  # Reset the BytesIO stream to the beginning
            image_data = image_input.read()
        else:
            raise ValueError("Unsupported input type. Must be a file path or BytesIO.")

        if self._validate_image_format(file_extension):
            enc_img = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/{file_extension};base64,{enc_img}"
        else:
            raise ValueError(f"Unsupported image format: {file_extension}")

    def _validate_image_format(self, file_type: str) -> bool:
        """Validate image format."""
        return file_type.lower() in SUPPORTED_IMAGE_FORMATS

    def _embed(
        self,
        texts: Optional[List[str]] = None,
        input_type: str = "search_document",
    ) -> List[List[float]]:
        """Embed sentences using Cohere."""
        client = self._get_client()

        if self.model_name not in (V3_MODELS + V4_MODELS):
            input_type = None
        else:
            input_type = self.input_type or input_type

        result = client.embed(
            texts=texts,
            input_type=input_type,
            embedding_types=[self.embedding_type],
            model=self.model_name,
            truncate=self.truncate,
        ).embeddings

        return getattr(result, self.embedding_type, None)

    async def _aembed(
        self,
        texts: Optional[List[str]] = None,
        input_type: str = "search_document",
    ) -> List[List[float]]:
        """Embed sentences using Cohere."""
        async_client = self._get_async_client()

        if self.model_name not in (V3_MODELS + V4_MODELS):
            input_type = None
        else:
            input_type = self.input_type or input_type

        result = (
            await async_client.embed(
                texts=texts,
                input_type=input_type,
                embedding_types=[self.embedding_type],
                model=self.model_name,
                truncate=self.truncate,
            )
        ).embeddings

        return getattr(result, self.embedding_type, None)

    def _embed_image(
        self, image_paths: List[ImageType], input_type: str
    ) -> List[List[float]]:
        """Embed images using Cohere."""
        if self.model_name not in (V3_MODELS + V4_MODELS):
            raise ValueError(
                f"{self.model_name} is not a valid multi-modal embedding model. Supported models are {V3_MODELS + V4_MODELS}"
            )

        client = self._get_client()
        processed_images = [
            self._image_to_base64_data_url(image_path) for image_path in image_paths
        ]

        inputs = [
            {"content": [{"type": "image_url", "image_url": {"url": processed_image}}]}
            for processed_image in processed_images
        ]

        embeddings = client.embed(
            inputs=inputs,
            input_type=input_type,
            embedding_types=[self.embedding_type],
            model=self.model_name,
            truncate=self.truncate,
        ).embeddings

        return getattr(embeddings, self.embedding_type, None)

    async def _aembed_image(
        self,
        image_paths: List[ImageType],
        input_type: str,
    ) -> List[List[float]]:
        """Embed images using Cohere."""
        if self.model_name not in (V3_MODELS + V4_MODELS):
            raise ValueError(
                f"{self.model_name} is not a valid multi-modal embedding model. Supported models are {V3_MODELS + V4_MODELS}"
            )

        async_client = self._get_async_client()
        processed_images = [
            self._image_to_base64_data_url(image_path) for image_path in image_paths
        ]

        inputs = [
            {"content": [{"type": "image_url", "image_url": {"url": processed_image}}]}
            for processed_image in processed_images
        ]

        embeddings = (
            await async_client.embed(
                inputs=inputs,
                input_type=input_type,
                embedding_types=[self.embedding_type],
                model=self.model_name,
                truncate=self.truncate,
            )
        ).embeddings

        return getattr(embeddings, self.embedding_type, None)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding. For query embeddings, input_type='search_query'."""
        return self._embed([query], input_type="search_query")[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async. For query embeddings, input_type='search_query'."""
        return (await self._aembed([query], input_type="search_query"))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text], input_type="search_document")[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return (await self._aembed([text], input_type="search_document"))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, input_type="search_document")

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return await self._aembed(texts, input_type="search_document")

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding."""
        return self._embed_image([img_file_path], "image")[0]

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding async."""
        return (await self._aembed_image([img_file_path], "image"))[0]

    def _get_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        """Get image embeddings."""
        return self._embed_image(img_file_paths, "image")

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        """Get image embeddings async."""
        return await self._aembed_image(img_file_paths, "image")
