"""Jina embeddings file."""

from typing import Any, List, Optional
from urllib.parse import urlparse
from os.path import exists
import base64
import requests
import numpy as np

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.schema import ImageType

MAX_BATCH_SIZE = 2048

DEFAULT_JINA_AI_API_URL = "https://api.jina.ai/v1"

VALID_ENCODING = ["float", "ubinary", "binary"]


class _JinaAPICaller:
    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        base_url: str = DEFAULT_JINA_AI_API_URL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.api_url = f"{base_url}/embeddings"
        self.api_key = get_from_param_or_env("api_key", api_key, "JINAAI_API_KEY", "")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"}
        )

    def get_embeddings(
        self,
        input,
        encoding_type: str = "float",
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = None,
    ) -> List[List[float]]:
        """Get embeddings."""
        # Call Jina AI Embedding API
        input_json = {
            "input": input,
            "model": self.model,
            "encoding_type": encoding_type,
        }
        if task is not None:
            input_json["task"] = task
        if dimensions is not None:
            input_json["dimensions"] = dimensions
        if late_chunking is not None:
            input_json["late_chunking"] = late_chunking

        resp = self._session.post(  # type: ignore
            self.api_url,
            json=input_json,
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        if encoding_type == "ubinary":
            return [
                np.unpackbits(np.array(result["embedding"], dtype="uint8")).tolist()
                for result in sorted_embeddings
            ]
        elif encoding_type == "binary":
            return [
                np.unpackbits(
                    (np.array(result["embedding"]) + 128).astype("uint8")
                ).tolist()
                for result in sorted_embeddings
            ]
        return [result["embedding"] for result in sorted_embeddings]

    async def aget_embeddings(
        self,
        input,
        encoding_type: str = "float",
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = None,
    ) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import aiohttp

        async with aiohttp.ClientSession(trust_env=True) as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept-Encoding": "identity",
            }
            input_json = {
                "input": input,
                "model": self.model,
                "encoding_type": encoding_type,
            }
            if task is not None:
                input_json["task"] = task
            if dimensions is not None:
                input_json["dimensions"] = dimensions
            if late_chunking is not None:
                input_json["late_chunking"] = late_chunking

            async with session.post(
                self.api_url,
                json=input_json,
                headers=headers,
            ) as response:
                resp = await response.json()
                response.raise_for_status()
                embeddings = resp["data"]

                # Sort resulting embeddings by index
                sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

                # Return just the embeddings
                if encoding_type == "ubinary":
                    return [
                        np.unpackbits(
                            np.array(result["embedding"], dtype="uint8")
                        ).tolist()
                        for result in sorted_embeddings
                    ]
                elif encoding_type == "binary":
                    return [
                        np.unpackbits(
                            (np.array(result["embedding"]) + 128).astype("uint8")
                        ).tolist()
                        for result in sorted_embeddings
                    ]
                return [result["embedding"] for result in sorted_embeddings]


def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def get_bytes_str(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class JinaEmbedding(MultiModalEmbedding):
    """
    JinaAI class for embeddings.

    Args:
        model (str): Model for embedding.
            Defaults to `jina-embeddings-v3`

    """

    api_key: Optional[str] = Field(default=None, description="The JinaAI API key.")
    model: str = Field(
        default="jina-embeddings-v3",
        description="The model to use when calling Jina AI API",
    )

    _encoding_queries: str = PrivateAttr()
    _encoding_documents: str = PrivateAttr()
    _task: str = PrivateAttr()
    _api: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        encoding_queries: Optional[str] = None,
        encoding_documents: Optional[str] = None,
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self._encoding_queries = encoding_queries or "float"
        self._encoding_documents = encoding_documents or "float"
        self._task = task
        self._dimensions = dimensions
        self._late_chunking = late_chunking

        assert self._encoding_documents in VALID_ENCODING, (
            f"Encoding Documents parameter {self._encoding_documents} not supported. Please choose one of {VALID_ENCODING}"
        )
        assert self._encoding_queries in VALID_ENCODING, (
            f"Encoding Queries parameter {self._encoding_documents} not supported. Please choose one of {VALID_ENCODING}"
        )

        self._api = _JinaAPICaller(model=model, api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "JinaAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._api.get_embeddings(
            input=[query],
            encoding_type=self._encoding_queries,
            task=self._task,
            dimensions=self._dimensions,
            late_chunking=self._late_chunking,
        )[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        result = await self._api.aget_embeddings(
            input=[query],
            encoding_type=self._encoding_queries,
            task=self._task,
            dimensions=self._dimensions,
            late_chunking=self._late_chunking,
        )
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._api.get_embeddings(
            input=texts,
            encoding_type=self._encoding_documents,
            task=self._task,
            dimensions=self._dimensions,
            late_chunking=self._late_chunking,
        )

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return await self._api.aget_embeddings(
            input=texts,
            encoding_type=self._encoding_documents,
            task=self._task,
            dimensions=self._dimensions,
            late_chunking=self._late_chunking,
        )

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        if is_local(img_file_path):
            input = [{"bytes": get_bytes_str(img_file_path)}]
        else:
            input = [{"url": img_file_path}]
        return self._api.get_embeddings(input=input)[0]

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        if is_local(img_file_path):
            input = [{"bytes": get_bytes_str(img_file_path)}]
        else:
            input = [{"url": img_file_path}]
        return await self._api.aget_embeddings(input=input)[0]

    def _get_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        input = []
        for img_file_path in img_file_paths:
            if is_local(img_file_path):
                input.append({"bytes": get_bytes_str(img_file_path)})
            else:
                input.append({"url": img_file_path})
        return self._api.get_embeddings(input=input)

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[List[float]]:
        input = []
        for img_file_path in img_file_paths:
            if is_local(img_file_path):
                input.append({"bytes": get_bytes_str(img_file_path)})
            else:
                input.append({"url": img_file_path})
        return await self._api.aget_embeddings(input=input)
