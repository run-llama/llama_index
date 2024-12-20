from typing import Callable, List, Optional, Union

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.utils.huggingface import format_query, format_text

DEFAULT_URL = "http://127.0.0.1:8080"


class TextEmbeddingsInference(BaseEmbedding):
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    truncate_text: bool = Field(
        default=True,
        description="Whether to truncate text or not when generating embeddings.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        text_instruction: Optional[str] = None,
        query_instruction: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        timeout: float = 60.0,
        truncate_text: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
    ):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            text_instruction=text_instruction,
            query_instruction=query_instruction,
            embed_batch_size=embed_batch_size,
            timeout=timeout,
            truncate_text=truncate_text,
            callback_manager=callback_manager,
            auth_token=auth_token,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingsInference"

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                auth_token = self.auth_token(self.base_url)
            else:
                auth_token = self.auth_token
            headers["Authorization"] = f"Bearer {auth_token}"

        json_data = {"inputs": texts, "truncate": self.truncate_text}

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/embed",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = format_query(query, self.model_name, self.query_instruction)
        return self._call_api([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = format_text(text, self.model_name, self.text_instruction)
        return self._call_api([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return self._call_api(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        query = format_query(query, self.model_name, self.query_instruction)
        return (await self._acall_api([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        text = format_text(text, self.model_name, self.text_instruction)
        return (await self._acall_api([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return await self._acall_api(texts)
