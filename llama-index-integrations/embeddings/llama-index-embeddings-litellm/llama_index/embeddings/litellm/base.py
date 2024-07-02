from typing import List
from litellm import embedding

from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings import BaseEmbedding


def get_embeddings(api_key: str, api_base: str, model_name: str, input: List[str]):
    if not api_key:
        # If key is not provided, we assume the consumer has configured
        # their LiteLLM proxy server with their API key.
        api_key = "some key"

    response = embedding(
        api_key=api_key,
        api_base=api_base,
        model=model_name,
        input=input,
    )
    return [result["embedding"] for result in response.data]


class LiteLLMEmbedding(BaseEmbedding):
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    api_key: str = Field(
        default="unknown",
        description="OpenAI key. If not provided, the proxy server must be configured with the key.",
    )
    api_base: str = Field(
        default="unknown", description="The base URL of the LiteLLM proxy."
    )

    @classmethod
    def class_name(cls) -> str:
        return "lite-llm"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            input=[query],
        )
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            input=[text],
        )
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            input=texts,
        )
