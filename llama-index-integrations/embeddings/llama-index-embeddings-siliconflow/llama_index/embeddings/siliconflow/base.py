"""SiliconFLow embeddings file."""

import aiohttp
import base64
import requests
import struct
from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding


DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"

VALID_ENCODING = ["float", "base64"]

AVAILABLE_OPTIONS = [
    ("Pro/BAAI/bge-m3", 1024),  ## 8192 tokens
    ("BAAI/bge-m3", 1024),  ## 8192 tokens
    ("BAAI/bge-large-zh-v1.5", 1024),  ## 512 tokens
    ("BAAI/bge-large-en-v1.5", 1024),  ## 512 tokens
    ("netease-youdao/bce-embedding-base_v1", 768),  ## 512 tokens
]


def base64_to_float_list(encoded_str: str) -> List[float]:
    byte_data = base64.b64decode(encoded_str)
    float_count = len(byte_data) // 4
    float_list = struct.unpack(f"{float_count}f", byte_data)
    return list(float_list)


class SiliconFlowEmbedding(BaseEmbedding):
    """SiliconFlow class for embeddings."""

    model: str = Field(
        default="BAAI/bge-m3",
        description="""\
            The name of the embedding model to use.
            512 tokens for all models input except `bge-m3` which is 8192.
        """,
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The SiliconFlow API key.",
    )
    base_url: str = Field(
        default=DEFAULT_SILICONFLOW_API_URL,
        description="The base URL for the SiliconFlow API.",
    )
    encoding_format: str = Field(
        default="float",
        description="The format to return the embeddings in. Can be either float or base64.",
    )  # TODO: Consider whether to fix the encoding format as float.

    _headers: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_SILICONFLOW_API_URL,
        encoding_format: Optional[str] = "float",
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            encoding_format=encoding_format,
            callback_manager=callback_manager,
            **kwargs,
        )
        assert (
            self.encoding_format in VALID_ENCODING
        ), f"""\
            Encoding_format parameter {self.encoding_format} not supported.
            Please choose one of {VALID_ENCODING}".
        """

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def class_name(cls) -> str:
        return "SiliconFlowEmbedding"

    def _data_formatting(self, response: list) -> List[List[float]]:
        results = sorted(response["data"], key=lambda e: e["index"])
        if self.encoding_format == "base64":
            return [base64_to_float_list(data["embedding"]) for data in results]
        else:
            return [data["embedding"] for data in results]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        result = await self._aget_text_embeddings([query])
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        with requests.Session() as session:
            input_json = {
                "model": self.model,
                "input": texts,
                "encoding_format": self.encoding_format,
            }
            response = session.post(
                self.base_url, json=input_json, headers=self._headers
            ).json()
            if "data" not in response:
                raise RuntimeError(response)
            return self._data_formatting(response)

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            input_json = {
                "input": texts,
                "model": self.model,
                "encoding_format": self.encoding_format,
            }

            async with session.post(
                self.base_url, json=input_json, headers=self._headers
            ) as response:
                response_json = await response.json()
                response.raise_for_status()
                return self._data_formatting(response_json)
