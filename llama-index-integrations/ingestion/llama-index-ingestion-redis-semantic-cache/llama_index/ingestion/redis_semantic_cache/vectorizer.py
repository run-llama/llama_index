from typing import Callable, List, Optional, Union, Any

from redisvl.utils.vectorize.base import BaseVectorizer

from llama_index.core.bridge.pydantic import (
    Field,
    ConfigDict,
)
from llama_index.core.base.embeddings.base import BaseEmbedding


class LlamaIndexVectorizer(BaseVectorizer):
    embed_model: BaseEmbedding = Field(
        ..., description="LlamaIndex embedding model to use for vectorization."
    )
    model: str = Field(..., description="Name of the embedding model.")
    dims: int = Field(..., description="Dimensionality of the embedding vectors.")
    dtype: str = Field(
        default="float32", description="Data type of the embedding vectors."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, embed_model: BaseEmbedding, embedding_dims: int, dtype: str = "float32"
    ):
        super().__init__(
            model="gemini-embedding-001",
            dims=embedding_dims,
            dtype=dtype,
            embed_model=embed_model,
        )

        self.embed_model = embed_model
        self.dtype = dtype

    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[float]:
        if isinstance(texts, str):
            return self.embed_model.get_text_embedding(texts)
        return self.embed_model._get_text_embeddings(texts)

    async def aencode(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[float]:
        if isinstance(texts, str):
            return await self.embed_model.aget_text_embedding(texts)
        return await self.embed_model._aget_text_embeddings(texts)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """
        Embed a single string.

        Args:
            text: The text to embed.
            preprocess: Optional callable applied to *text* before embedding.
            as_buffer: Return raw bytes instead of a list of floats.

        Returns:
            Embedding as ``List[float]``, or ``bytes`` when *as_buffer* is ``True``.

        """
        if not isinstance(text, str):
            raise TypeError("embed() requires a str value.")

        if preprocess:
            text = preprocess(text)

        embedding = self.encode(text, **kwargs)
        return self._process_embedding(embedding, as_buffer, self.dtype)

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """
        Embed a list of strings synchronously.

        Args:
            texts: Strings to embed.
            preprocess: Optional callable applied to each string before embedding.
            as_buffer: Return raw bytes instead of lists of floats.

        Returns:
            List of embeddings (each a ``List[float]`` or ``bytes``).

        """
        if not isinstance(texts, list):
            raise TypeError("embed_many() requires a list of str values.")

        if preprocess:
            texts = [preprocess(t) for t in texts]

        embeddings: List[List[float]] = []
        for batch in self.batchify(texts, self.embed_model.embed_batch_size):
            embeddings.extend(self.encode(batch))

        return [self._process_embedding(e, as_buffer, self.dtype) for e in embeddings]

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """
        Async embed a single string.

        Args:
            text: The text to embed.
            preprocess: Optional callable applied to *text* before embedding.
            as_buffer: Return raw bytes instead of a list of floats.

        Returns:
            Embedding as ``List[float]``, or ``bytes`` when *as_buffer* is ``True``.

        """
        if not isinstance(text, str):
            raise TypeError("aembed() requires a str value.")

        if preprocess:
            text = preprocess(text)

        embedding = await self.aencode(text, **kwargs)
        return self._process_embedding(embedding, as_buffer, self.dtype)

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """
        Async embed a list of strings.

        Args:
            texts: Strings to embed.
            preprocess: Optional callable applied to each string before embedding.
            as_buffer: Return raw bytes instead of lists of floats.

        Returns:
            List of embeddings (each a ``List[float]`` or ``bytes``).

        """
        if not isinstance(texts, list):
            raise TypeError("aembed_many() requires a list of str values.")

        if preprocess:
            texts = [preprocess(t) for t in texts]

        embeddings: List[List[float]] = []
        for batch in self.batchify(texts, self.embed_model.embed_batch_size):
            embeddings.extend(await self.aencode(batch))

        return [self._process_embedding(e, as_buffer, self.dtype) for e in embeddings]
