"""TwelveLabs Marengo multimodal embeddings."""

from __future__ import annotations

from os.path import exists
from typing import Any, List, Optional
from urllib.parse import urlparse

import requests
from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.schema import ImageType

API_BASE = "https://api.twelvelabs.io/v1.3"
DEFAULT_MODEL = "marengo3.0"


class TwelveLabsEmbedding(MultiModalEmbedding):
    """
    TwelveLabs Marengo multimodal embeddings (text + image) in a shared space.

    Marengo embeds text, images, audio, and video into one vector space, which
    enables cross-modal retrieval (e.g. text-to-image search). This integration
    exposes Marengo's synchronous text and image embeddings through LlamaIndex's
    ``MultiModalEmbedding`` interface; both return vectors that are directly
    comparable with cosine similarity.

    Set the ``TWELVELABS_API_KEY`` environment variable (or pass ``api_key``).
    Get a key at https://playground.twelvelabs.io.

    Note:
        Marengo also embeds full videos, but a video produces multiple
        time-segment vectors (an async task) rather than the single vector
        LlamaIndex's embedding interface expects, so video embedding is out of
        scope here. Use ``llama-index-readers-twelvelabs`` (Pegasus) to turn a
        video into text Documents.

    Example:
        >>> from llama_index.embeddings.twelvelabs import TwelveLabsEmbedding
        >>> embed_model = TwelveLabsEmbedding()
        >>> vector = embed_model.get_text_embedding("a cat playing piano")

    """

    _api_key: str = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        key = get_from_param_or_env("api_key", api_key, "TWELVELABS_API_KEY", "")
        if not key:
            raise ValueError(
                "A TwelveLabs API key is required. Pass api_key=... or set the "
                "TWELVELABS_API_KEY environment variable."
            )
        self._api_key = key
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": key})

    @classmethod
    def class_name(cls) -> str:
        return "TwelveLabsEmbedding"

    # -- TwelveLabs /embed (multipart/form-data) ---------------------------- #
    def _embed(self, fields: dict) -> List[float]:
        response = self._session.post(f"{API_BASE}/embed", files=fields, timeout=120)
        if not response.ok:
            raise RuntimeError(
                f"TwelveLabs embed failed: HTTP {response.status_code} "
                f"{response.text[:300]}"
            )
        return _first_vector(response.json())

    def _embed_text(self, text: str) -> List[float]:
        # `(None, value)` tuples send text fields as multipart/form-data.
        return self._embed(
            {"model_name": (None, self.model_name), "text": (None, text)}
        )

    def _embed_image(self, image: ImageType) -> List[float]:
        if isinstance(image, str) and urlparse(image).scheme in ("http", "https"):
            return self._embed(
                {"model_name": (None, self.model_name), "image_url": (None, image)}
            )
        if isinstance(image, str):
            if not exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            with open(image, "rb") as handle:
                data = handle.read()
        elif hasattr(image, "read"):
            data = image.read()
        else:
            data = bytes(image)
        return self._embed(
            {"model_name": (None, self.model_name), "image_file": ("image", data)}
        )

    # -- MultiModalEmbedding interface -------------------------------------- #
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_text(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        return self._embed_image(img_file_path)

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        return self._get_image_embedding(img_file_path)


def _first_vector(data: Any) -> List[float]:
    """Extract the embedding float vector from a Marengo /embed response."""
    if isinstance(data, dict):
        for key in (
            "text_embedding",
            "image_embedding",
            "audio_embedding",
            "video_embedding",
        ):
            embedding = data.get(key)
            if isinstance(embedding, dict):
                segments = embedding.get("segments") or []
                if segments and isinstance(segments[0], dict):
                    vector = segments[0].get("float")
                    if isinstance(vector, list):
                        return [float(x) for x in vector]
    raise RuntimeError(f"TwelveLabs embed: no vector in response: {str(data)[:200]}")
