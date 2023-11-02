from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding

# Constants for default values
DEFAULT_MODELS_REQUIRING_INPUT_TYPE = [
    "embed-english-v3.0",
    "embed-english-light-3.0",
    "embed-multilingual-v3.0",
    "embed-multilingual-light-v3.0",
]

DEFAULT_INPUT_TYPES = [
    "search_query",
    "search_documents",
    "classification",
    "clustering",
]
DEFAULT_TRUNCATE_OPTIONS = ["START", "END", "NONE"]


# Assuming BaseEmbedding is a Pydantic model and handles its own initializations
class CohereEmbedding(BaseEmbedding):
    """CohereEmbedding uses the Cohere API to generate embeddings for text."""

    # Instance variables initialized via Pydantic's mechanism
    cohere_client: Any = Field(description="CohereAI client")
    model_name: str = Field(description="CohereAI model name")
    truncate: str = Field(description="Truncation type - START/ END/ NONE")
    input_type: Optional[str] = Field(description="Model Input type")

    def __init__(
        self,
        cohere_api_key: str = None,
        model_name: str = "embed-english-v2.0",
        truncate: str = "END",
        input_type: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        # Attempt to import cohere. If it fails, raise an informative ImportError.
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "CohereEmbedding requires the 'cohere' package to be installed.\n"
                "Please install it with `pip install cohere`."
            )
        # Validation for model and input_type
        if model_name in DEFAULT_MODELS_REQUIRING_INPUT_TYPE:
            if input_type not in DEFAULT_INPUT_TYPES:
                raise ValueError(
                    f"input_type must be one of {DEFAULT_INPUT_TYPES} for model '{model_name}'"
                )
        else:
            if input_type is not None:
                raise ValueError(
                    f"input_type should not be provided for model '{model_name}'"
                )

        if truncate not in DEFAULT_TRUNCATE_OPTIONS:
            raise ValueError(f"truncate must be one of {DEFAULT_TRUNCATE_OPTIONS}")

        super().__init__(
            cohere_client=cohere.Client(cohere_api_key),
            cohere_api_key=cohere_api_key,
            model_name=model_name,
            truncate=truncate,
            input_type=input_type,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CohereEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed sentences using Cohere."""
        if self.input_type:
            embeddings = self.cohere_client.embed(
                texts=texts,
                input_type=self.input_type,
                model=self.model_name,
                truncate=self.truncate,
            ).embeddings
        else:
            embeddings = self.cohere_client.embed(
                texts=texts, model=self.model_name, truncate=self.truncate
            ).embeddings
        return [list(map(float, e)) for e in embeddings]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts)
