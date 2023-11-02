from typing import List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.embeddings.huggingface_utils import format_query, format_text

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

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        text_instruction: Optional[str] = None,
        query_instruction: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            import httpx  # noqa
        except ImportError:
            raise ImportError(
                "TextEmbeddingsInterface requires httpx to be installed.\n"
                "Please install httpx with `pip install httpx`."
            )

        super().__init__(
            base_url=base_url,
            model_name=model_name,
            text_instruction=text_instruction,
            query_instruction=query_instruction,
            embed_batch_size=embed_batch_size,
            timeout=timeout,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingsInference"

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"inputs": texts}

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
        json_data = {"inputs": texts}

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
