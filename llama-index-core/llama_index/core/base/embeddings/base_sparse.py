"""Base sparse embeddings file."""

import asyncio
import math
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    model_serializer,
)
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.embedding import (
    SparseEmbeddingEndEvent,
    SparseEmbeddingStartEvent,
)
import llama_index.core.instrumentation as instrument
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.async_utils import run_jobs


dispatcher = instrument.get_dispatcher(__name__)

SparseEmbedding = Dict[int, float]


def sparse_similarity(
    embedding1: SparseEmbedding,
    embedding2: SparseEmbedding,
) -> float:
    """Get sparse embedding similarity."""
    if not embedding1 or not embedding2:
        return 0.0

    # Use the smaller embedding as the primary iteration set
    if len(embedding1) > len(embedding2):
        embedding1, embedding2 = embedding2, embedding1

    # Precompute norms and find common indices
    norm1 = norm2 = dot_product = 0.0
    common_indices = set(embedding1.keys()) & set(embedding2.keys())

    for idx, value in embedding1.items():
        norm1 += value**2
        if idx in common_indices:
            dot_product += value * embedding2[idx]

    for value in embedding2.values():
        norm2 += value**2

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))


def mean_agg(embeddings: List[SparseEmbedding]) -> SparseEmbedding:
    """Get mean aggregation of embeddings."""
    if not embeddings:
        return {}

    sum_dict: Dict[int, float] = defaultdict(float)
    for embedding in embeddings:
        for idx, value in embedding.items():
            sum_dict[idx] += value

    return {idx: value / len(embeddings) for idx, value in sum_dict.items()}


class BaseSparseEmbedding(BaseModel, DispatcherSpanMixin):
    """Base class for embeddings."""

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "BaseSparseEmbedding"

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> Dict[str, Any]:
        data = handler(self)
        # add class name
        data["class_name"] = self.class_name()
        # del api_key if it exists
        data.pop("api_key", None)
        return data

    @abstractmethod
    def _get_query_embedding(self, query: str) -> SparseEmbedding:
        """Embed the input query synchronously."""

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> SparseEmbedding:
        """Embed the input query asynchronously."""

    @dispatcher.span
    def get_query_embedding(self, query: str) -> SparseEmbedding:
        """Embed the input query."""
        model_dict = self.model_dump()
        dispatcher.event(
            SparseEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )

        query_embedding = self._get_query_embedding(query)

        dispatcher.event(
            SparseEmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    @dispatcher.span
    async def aget_query_embedding(self, query: str) -> SparseEmbedding:
        """Get query embedding."""
        model_dict = self.model_dump()
        dispatcher.event(
            SparseEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )

        query_embedding = await self._aget_query_embedding(query)

        dispatcher.event(
            SparseEmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., SparseEmbedding]] = None,
    ) -> SparseEmbedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., SparseEmbedding]] = None,
    ) -> SparseEmbedding:
        """Async get aggregated embedding from multiple queries."""
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> SparseEmbedding:
        """Embed the input text synchronously."""

    @abstractmethod
    async def _aget_text_embedding(self, text: str) -> SparseEmbedding:
        """Embed the input text asynchronously."""

    def _get_text_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        """
        Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        """
        Embed the input sequence of text asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    @dispatcher.span
    def get_text_embedding(self, text: str) -> SparseEmbedding:
        """Embed the input text."""
        model_dict = self.model_dump()
        dispatcher.event(
            SparseEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )

        text_embedding = self._get_text_embedding(text)

        dispatcher.event(
            SparseEmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    async def aget_text_embedding(self, text: str) -> SparseEmbedding:
        """Async get text embedding."""
        model_dict = self.model_dump()
        dispatcher.event(
            SparseEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )

        text_embedding = await self._aget_text_embedding(text)

        dispatcher.event(
            SparseEmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    def get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[SparseEmbedding]:
        """Get a list of text embeddings, with batching."""
        cur_batch: List[str] = []
        result_embeddings: List[SparseEmbedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        model_dict = self.model_dump()
        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    SparseEmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )

                embeddings = self._get_text_embeddings(cur_batch)
                result_embeddings.extend(embeddings)

                dispatcher.event(
                    SparseEmbeddingEndEvent(
                        chunks=cur_batch,
                        embeddings=embeddings,
                    )
                )
                cur_batch = []

        return result_embeddings

    @dispatcher.span
    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[SparseEmbedding]:
        """Asynchronously get a list of text embeddings, with batching."""
        num_workers = self.num_workers

        model_dict = self.model_dump()

        cur_batch: List[str] = []
        callback_payloads: List[List[str]] = []
        result_embeddings: List[SparseEmbedding] = []
        embeddings_coroutines: List[Coroutine] = []
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    SparseEmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )

                callback_payloads.append(cur_batch)
                embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []

        if num_workers and num_workers > 1:
            nested_embeddings = await run_jobs(
                embeddings_coroutines,
                show_progress=show_progress,
                workers=self.num_workers,
                desc="Generating embeddings",
            )
        else:
            if show_progress:
                try:
                    from tqdm.asyncio import tqdm_asyncio

                    nested_embeddings = await tqdm_asyncio.gather(
                        *embeddings_coroutines,
                        total=len(embeddings_coroutines),
                        desc="Generating embeddings",
                    )
                except ImportError:
                    nested_embeddings = await asyncio.gather(*embeddings_coroutines)
            else:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for text_batch, embeddings in zip(callback_payloads, nested_embeddings):
            dispatcher.event(
                SparseEmbeddingEndEvent(
                    chunks=text_batch,
                    embeddings=embeddings,
                )
            )

        return result_embeddings

    def similarity(
        self,
        embedding1: SparseEmbedding,
        embedding2: SparseEmbedding,
    ) -> float:
        """Get sparse embedding similarity."""
        return sparse_similarity(embedding1, embedding2)
