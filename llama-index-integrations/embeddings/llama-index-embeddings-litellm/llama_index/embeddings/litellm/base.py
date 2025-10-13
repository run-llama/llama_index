from typing import Any, List, Optional
from litellm import embedding

from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings import BaseEmbedding


def get_embeddings(
    api_key: str,
    api_base: str,
    model_name: str,
    input: List[str],
    timeout: int = 60,
    **kwargs: Any,
) -> List[List[float]]:
    """
    Retrieve embeddings for a given list of input strings using the specified model.

    Args:
        api_key (str): The API key for authentication.
        api_base (str): The base URL of the LiteLLM proxy server.
        model_name (str): The name of the model to use for generating embeddings.
        input (List[str]): A list of input strings for which embeddings are to be generated.
        timeout (float): The timeout value for the API call, default 60 secs.
        **kwargs (Any): Additional keyword arguments to be passed to the embedding function.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding corresponds to an input string.

    """
    response = embedding(
        api_key=api_key,
        api_base=api_base,
        model=model_name,
        input=input,
        timeout=timeout,
        **kwargs,
    )
    return [result["embedding"] for result in response.data]


class LiteLLMEmbedding(BaseEmbedding):
    model_name: str = Field(description="The name of the embedding model.")
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI key. If not provided, the proxy server must be configured with the key.",
    )
    api_base: Optional[str] = Field(
        default=None, description="The base URL of the LiteLLM proxy."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in text-embedding-3 and later models."
        ),
    )
    timeout: Optional[int] = Field(
        default=60, description="Timeout for each request.", ge=0
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
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=[query],
        )
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=[text],
        )
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=texts,
        )
