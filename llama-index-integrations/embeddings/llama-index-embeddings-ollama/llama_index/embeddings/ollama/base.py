import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE

from ollama import Client, AsyncClient


class OllamaEmbedding(BaseEmbedding):
    """Class for Ollama embeddings."""

    base_url: str = Field(description="Base url the model is hosted by Ollama")
    model_name: str = Field(description="The Ollama model to use.")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    ollama_additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )
    query_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend to text."
    )

    _client: Client = PrivateAttr()
    _async_client: AsyncClient = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        ollama_additional_kwargs: Optional[Dict[str, Any]] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            callback_manager=callback_manager,
            **kwargs,
        )

        client_kwargs = client_kwargs or {}
        self._client = Client(host=self.base_url, **client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **client_kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        formatted_query = self._format_query(query)
        return self.get_general_text_embedding(formatted_query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        formatted_query = self._format_query(query)
        return await self.aget_general_text_embedding(formatted_query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        formatted_text = self._format_text(text)
        return self.get_general_text_embedding(formatted_text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        formatted_text = self._format_text(text)
        return await self.aget_general_text_embedding(formatted_text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            formatted_text = self._format_text(text)
            embeddings = self.get_general_text_embedding(formatted_text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        formatted_texts = [self._format_text(text) for text in texts]
        return await asyncio.gather(
            *[self.aget_general_text_embedding(text) for text in formatted_texts]
        )

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        result = self._client.embed(
            model=self.model_name, input=texts, options=self.ollama_additional_kwargs
        )
        return result.embeddings[0]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embed(
            model=self.model_name, input=prompt, options=self.ollama_additional_kwargs
        )
        return result.embeddings[0]

    def _format_query(self, query: str) -> str:
        """Format query with instruction if provided."""
        if self.query_instruction:
            return f"{self.query_instruction.strip()} {query.strip()}".strip()
        return query.strip()

    def _format_text(self, text: str) -> str:
        """Format text with instruction if provided."""
        if self.text_instruction:
            return f"{self.text_instruction.strip()} {text.strip()}".strip()
        return text.strip()
