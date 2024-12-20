from typing import Any, List, Optional, Dict

from aleph_alpha_client import (
    Client,
    AsyncClient,
    Prompt,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
    BatchSemanticEmbeddingRequest,
    BatchSemanticEmbeddingResponse,
)
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.bridge.pydantic import Field, PrivateAttr

DEFAULT_ALEPHALPHA_MODEL = "luminous-base"
DEFAULT_ALEPHALPHA_HOST = "https://api.aleph-alpha.com"

VALID_REPRESENTATION_TYPES = [
    None,
    SemanticRepresentation.Symmetric,
    SemanticRepresentation.Query,
    SemanticRepresentation.Document,
]


class AlephAlphaEmbedding(BaseEmbedding):
    """AlephAlphaEmbedding uses the Aleph Alpha API to generate embeddings for text."""

    model: str = Field(
        default=DEFAULT_ALEPHALPHA_MODEL, description="The Aleph Alpha model to use."
    )
    token: str = Field(default=None, description="The Aleph Alpha API token.")
    representation: Optional[str] = Field(
        default=SemanticRepresentation.Query,
        description="The representation type to use for generating embeddings.",
    )
    compress_to_size: Optional[int] = Field(
        default=None,
        description="The size to compress the embeddings to.",
        gt=0,
    )
    base_url: Optional[str] = Field(
        default=DEFAULT_ALEPHALPHA_HOST, description="The hostname of the API base_url."
    )
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )
    normalize: Optional[bool] = Field(
        default=False, description="Return normalized embeddings."
    )
    hosting: Optional[str] = Field(default=None, description="The hosting to use.")
    nice: bool = Field(default=False, description="Whether to be nice to the API.")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Aleph Alpha API."
    )

    # Instance variables initialized via Pydantic's mechanism
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ALEPHALPHA_MODEL,
        token: Optional[str] = None,
        representation: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_ALEPHALPHA_HOST,
        hosting: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        nice: bool = False,
        verify_ssl: bool = True,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        A class representation for generating embeddings using the AlephAlpha API.

        Args:
            token: The token to use for the AlephAlpha API.
            model: The model to use for generating embeddings.
            base_url: The base URL of the AlephAlpha API.
            nice: Whether to use the "nice" mode for the AlephAlpha API.
            additional_kwargs: Additional kwargs for the AlephAlpha API.

        """
        additional_kwargs = additional_kwargs or {}

        super().__init__(
            model=model,
            representation=representation,
            base_url=base_url,
            token=token,
            nice=nice,
            additional_kwargs=additional_kwargs,
        )

        self.token = get_from_param_or_env("aa_token", token, "AA_TOKEN", "")

        if representation is not None and isinstance(representation, str):
            try:
                representation_enum = SemanticRepresentation[
                    representation.capitalize()
                ]
            except KeyError:
                raise ValueError(
                    f"{representation} is not a valid representation type. Available types are: {list(SemanticRepresentation.__members__.keys())}"
                )
            self.representation = representation_enum
        else:
            self.representation = representation

        self._client = None
        self._aclient = None

    @classmethod
    def class_name(cls) -> str:
        return "AlephAlphaEmbedding"

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "host": self.base_url,
            "hosting": self.hosting,
            "request_timeout_seconds": self.timeout,
            "total_retries": self.max_retries,
            "nice": self.nice,
            "verify_ssl": self.verify_ssl,
        }

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncClient:
        if self._aclient is None:
            self._aclient = AsyncClient(**self._get_credential_kwargs())
        return self._aclient

    def _get_embedding(self, text: str, representation: str) -> List[float]:
        """Embed sentence using AlephAlpha."""
        client = self._get_client()
        request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text),
            representation=representation or self.representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
        )
        result = client.semantic_embed(request=request, model=self.model)
        return result.embedding

    async def _aget_embedding(self, text: str, representation: str) -> List[float]:
        """Get embedding async."""
        aclient = self._get_aclient()
        request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text),
            representation=representation or self.representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
        )
        result = await aclient.semantic_embed(request=request, model=self.model)
        return result.embedding

    def _get_embeddings(
        self, texts: List[str], representation: str
    ) -> List[List[float]]:
        """Embed sentences using AlephAlpha."""
        client = self._get_client()
        request = BatchSemanticEmbeddingRequest(
            prompts=[Prompt.from_text(text) for text in texts],
            representation=representation or self.representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
        )
        result: BatchSemanticEmbeddingResponse = client.batch_semantic_embed(
            request=request, model=self.model
        )
        return result.embeddings

    async def _aget_embeddings(
        self, texts: List[str], representation: str
    ) -> List[List[float]]:
        """Get embeddings async."""
        aclient = self._get_aclient()
        request = BatchSemanticEmbeddingRequest(
            prompts=[Prompt.from_text(text) for text in texts],
            representation=representation or self.representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
        )
        result: BatchSemanticEmbeddingResponse = await aclient.batch_semantic_embed(
            request=request, model=self.model
        )
        return result.embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding. For query embeddings, representation='query'."""
        return self._get_embedding(query, SemanticRepresentation.Query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async. For query embeddings, representation='query'."""
        return self._aget_embedding(query, SemanticRepresentation.Query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding. For text embeddings, representation='document'."""
        return self._get_embedding(text, SemanticRepresentation.Document)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._aget_embedding(text, SemanticRepresentation.Document)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._get_embeddings(texts, SemanticRepresentation.Document)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings async."""
        return self._aget_embeddings(texts, SemanticRepresentation.Document)
