import asyncio
from typing import Any, List, Optional
from zhipuai import ZhipuAI as ZhipuAIClient
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager


class ZhipuAIEmbedding(BaseEmbedding):
    """ZhipuAI LLM.

    Visit https://open.bigmodel.cn to get more information about ZhipuAI.

    Examples:
        `pip install llama-index-embeddings-zhipuai`

        ```python
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

        embedding = ZhipuAIEmbedding(model="embedding-2", api_key="YOUR API KEY")

        response = embedding.get_general_text_embedding("who are you?")
        print(response)
        ```
    """

    model: str = Field(description="The ZhipuAI model to use.")
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the ZhipuAI API.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout to use for the ZhipuAI API.",
    )
    _client: Optional[ZhipuAIClient] = PrivateAttr()

    def __init__(
        self,
        model: str,
        api_key: str,
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = ZhipuAIClient(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "ZhipuAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.aget_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.aget_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)
        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await asyncio.gather(
            *[self.aget_general_text_embedding(text) for text in texts]
        )

    def get_general_text_embedding(self, text: str) -> List[float]:
        """Get ZhipuAI embeddings."""
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            timeout=self.timeout,
        )
        return response.data[0].embedding

    async def aget_general_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get ZhipuAI embeddings."""
        response = await asyncio.to_thread(
            self._client.embeddings.create,
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            timeout=self.timeout,
        )
        return response.data[0].embedding
