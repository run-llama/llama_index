"""Base embeddings file."""

import asyncio
import uuid
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional, Sequence, Tuple, cast
from typing_extensions import Self

import numpy as np
from llama_index.core.bridge.pydantic import (
    Field,
    ConfigDict,
    model_validator,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.async_utils import run_jobs
from llama_index.core.embeddings.mixed_embedding_utils import MixedEmbeddingContent

# TODO: change to numpy array
Embedding = List[float]


from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def mean_agg(embeddings: List[Embedding]) -> Embedding:
    """Mean aggregation for embeddings."""
    if not embeddings:
        raise ValueError("No embeddings to aggregate")

    return np.array(embeddings).mean(axis=0).tolist()


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


class BaseEmbedding(TransformComponent, DispatcherSpanMixin):
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
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )
    # Use Any to avoid import loops
    embeddings_cache: Optional[Any] = Field(
        default=None,
        description="Cache for the embeddings: if None, the embeddings are not cached",
    )
    # Expected type: BaseRateLimiter (from llama_index.core.rate_limiter)
    rate_limiter: Optional[Any] = Field(
        default=None,
        description="Rate limiter instance to throttle API calls.",
        exclude=True,
    )

    @property
    def supports_mixed_embedding(self) -> bool:
        """
        Whether this embedding model supports joint embedding of interleaved
        text and image content (mixed multimodal embedding).
        When True, nodes with mixed content will use embed().
        """
        return False

    def _get_mixed_content_embedding(self, content: MixedEmbeddingContent) -> Embedding:
        """
        Embed a single mixed content (interleaved text / image / audio / video) input.

        Subclasses that support mixed embedding should override this.
        Default raises NotImplementedError.
        """
        raise NotImplementedError(
            "This embedding model does not support mixed (interleaved) multimodal embedding."
        )

    def _get_mixed_content_embeddings(
        self, contents: List[MixedEmbeddingContent]
    ) -> List[Embedding]:
        """
        Embed a batch of mixed content inputs.

        Subclasses can override to use a native batch API when available.
        Default implementation calls _get_mixed_content_embedding for each item.
        """
        return [self._get_mixed_content_embedding(content) for content in contents]

    async def _aget_mixed_content_embedding(
        self, content: MixedEmbeddingContent
    ) -> Embedding:
        """
        Async embed a single mixed content input.

        Subclasses can override for true async implementation.
        Default falls back to sync _get_mixed_content_embedding.
        """
        return self._get_mixed_content_embedding(content)

    async def _aget_mixed_content_embeddings(
        self, contents: List[MixedEmbeddingContent]
    ) -> List[Embedding]:
        """
        Async embed a batch of mixed content inputs.

        Subclasses can override to use a native batch API when available.
        Default implementation gathers _aget_mixed_content_embedding per item.
        """
        return await asyncio.gather(
            *[self._aget_mixed_content_embedding(c) for c in contents]
        )

    @dispatcher.span
    def embed(self, content: MixedEmbeddingContent) -> Embedding:
        """Embed interleaved text + image/content (single item)."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(EmbeddingStartEvent(model_dict=model_dict))
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        embedding = self._get_mixed_content_embedding(content)
        dispatcher.event(EmbeddingEndEvent(chunks=[content], embeddings=[embedding]))
        return embedding

    @dispatcher.span
    async def aembed(self, content: MixedEmbeddingContent) -> Embedding:
        """Async embed interleaved text + image/content."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(EmbeddingStartEvent(model_dict=model_dict))
        if self.rate_limiter is not None:
            await self.rate_limiter.async_acquire()
        embedding = await self._aget_mixed_content_embedding(content)
        dispatcher.event(EmbeddingEndEvent(chunks=[content], embeddings=[embedding]))
        return embedding

    @dispatcher.span
    def embed_batch(
        self,
        contents: List[MixedEmbeddingContent],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[Embedding]:
        """Embed a list of mixed contents, with batching."""
        result_embeddings: List[Embedding] = []
        queue_with_progress = enumerate(
            get_tqdm_iterable(
                contents, show_progress, "Generating mixed content embeddings"
            )
        )
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        cur_batch: List[MixedEmbeddingContent] = []
        for idx, content in queue_with_progress:
            cur_batch.append(content)
            if idx == len(contents) - 1 or len(cur_batch) == self.embed_batch_size:
                dispatcher.event(EmbeddingStartEvent(model_dict=model_dict))
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                embeddings = self._get_mixed_content_embeddings(cur_batch)
                result_embeddings.extend(embeddings)
                dispatcher.event(
                    EmbeddingEndEvent(chunks=cur_batch, embeddings=embeddings)
                )
                cur_batch = []
        return result_embeddings

    @dispatcher.span
    async def aembed_batch(
        self,
        contents: List[MixedEmbeddingContent],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[Embedding]:
        """Async embed a list of mixed contents, with batching."""
        result_embeddings: List[Embedding] = []
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        cur_batch: List[MixedEmbeddingContent] = []
        embeddings_coroutines: List[Coroutine] = []
        batch_payloads: List[List[MixedEmbeddingContent]] = []
        for idx, content in enumerate(contents):
            cur_batch.append(content)
            if idx == len(contents) - 1 or len(cur_batch) == self.embed_batch_size:
                dispatcher.event(EmbeddingStartEvent(model_dict=model_dict))
                if self.rate_limiter is not None:
                    await self.rate_limiter.async_acquire()
                embeddings_coroutines.append(
                    self._aget_mixed_content_embeddings(cur_batch)
                )
                batch_payloads.append(cur_batch)
                cur_batch = []
        if embeddings_coroutines:
            nested = await asyncio.gather(*embeddings_coroutines)
            for embeddings in nested:
                result_embeddings.extend(embeddings)
            for batch, embeddings in zip(batch_payloads, nested):
                dispatcher.event(EmbeddingEndEvent(chunks=batch, embeddings=embeddings))
        return result_embeddings

    @model_validator(mode="after")
    def check_base_embeddings_class(self) -> Self:
        from llama_index.core.storage.kvstore.types import BaseKVStore

        if self.callback_manager is None:
            self.callback_manager = CallbackManager([])
        if self.embeddings_cache is not None and not isinstance(
            self.embeddings_cache, BaseKVStore
        ):
            raise TypeError("embeddings_cache must be of type BaseKVStore")
        return self

    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    @dispatcher.span
    def get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query.

        When embedding a query, depending on the model, a special instruction
        can be prepended to the raw query string. For example, "Represent the
        question for retrieving supporting documents: ". If you're curious,
        other examples of predefined instructions can be found in
        embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                query_embedding = self._get_query_embedding(query)
            elif self.embeddings_cache is not None:
                cached_emb = self.embeddings_cache.get(
                    key=query, collection="embeddings"
                )
                if cached_emb is not None:
                    cached_key = next(iter(cached_emb.keys()))
                    query_embedding = cached_emb[cached_key]
                else:
                    if self.rate_limiter is not None:
                        self.rate_limiter.acquire()
                    query_embedding = self._get_query_embedding(query)
                    self.embeddings_cache.put(
                        key=query,
                        val={str(uuid.uuid4()): query_embedding},
                        collection="embeddings",
                    )
            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    @dispatcher.span
    async def aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    await self.rate_limiter.async_acquire()
                query_embedding = await self._aget_query_embedding(query)
            elif self.embeddings_cache is not None:
                cached_emb = await self.embeddings_cache.aget(
                    key=query, collection="embeddings"
                )
                if cached_emb is not None:
                    cached_key = next(iter(cached_emb.keys()))
                    query_embedding = cached_emb[cached_key]
                else:
                    if self.rate_limiter is not None:
                        await self.rate_limiter.async_acquire()
                    query_embedding = await self._aget_query_embedding(query)
                    await self.embeddings_cache.aput(
                        key=query,
                        val={str(uuid.uuid4()): query_embedding},
                        collection="embeddings",
                    )

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., Embedding]] = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., Embedding]] = None,
    ) -> Embedding:
        """Async get aggregated embedding from multiple queries."""
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.

        Subclasses should implement this method. Reference get_text_embedding's
        docstring for more information.
        """

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text asynchronously.

        Subclasses can implement this method if there is a true async
        implementation. Reference get_text_embedding's docstring for more
        information.
        """
        # Default implementation just falls back on _get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    async def _aget_text_embeddings_rate_limited(
        self, texts: List[str]
    ) -> List[Embedding]:
        """Acquire rate limiter before delegating to _aget_text_embeddings."""
        if self.rate_limiter is not None:
            await self.rate_limiter.async_acquire()
        return await self._aget_text_embeddings(texts)

    def _get_text_embeddings_cached(self, texts: List[str]) -> List[Embedding]:
        """
        Get text embeddings from cache. If not in cache, generate them.
        """
        if self.embeddings_cache is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: List[Optional[Embedding]] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: List[Tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cached_emb = self.embeddings_cache.get(key=txt, collection="embeddings")
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                embeddings[i] = cached_emb[cached_key]
            else:
                non_cached_texts.append((i, txt))
        if len(non_cached_texts) > 0:
            text_embeddings = self._get_text_embeddings(
                [x[1] for x in non_cached_texts]
            )
            for j, text_embedding in enumerate(text_embeddings):
                orig_i = non_cached_texts[j][0]
                embeddings[orig_i] = text_embedding

                self.embeddings_cache.put(
                    key=texts[orig_i],
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return cast(List[Embedding], embeddings)

    async def _aget_text_embeddings_cached(self, texts: List[str]) -> List[Embedding]:
        """
        Asynchronously get text embeddings from cache. If not in cache, generate them.
        """
        if self.embeddings_cache is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: List[Optional[Embedding]] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: List[Tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cached_emb = await self.embeddings_cache.aget(
                key=txt, collection="embeddings"
            )
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                embeddings[i] = cached_emb[cached_key]
            else:
                non_cached_texts.append((i, txt))

        if len(non_cached_texts) > 0:
            text_embeddings = await self._aget_text_embeddings(
                [x[1] for x in non_cached_texts]
            )
            for j, text_embedding in enumerate(text_embeddings):
                orig_i = non_cached_texts[j][0]
                embeddings[orig_i] = text_embedding
                await self.embeddings_cache.aput(
                    key=texts[orig_i],
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return cast(List[Embedding], embeddings)

    @dispatcher.span
    def get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text.

        When embedding text, depending on the model, a special instruction
        can be prepended to the raw text string. For example, "Represent the
        document for retrieval: ". If you're curious, other examples of
        predefined instructions can be found in embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                text_embedding = self._get_text_embedding(text)
            elif self.embeddings_cache is not None:
                cached_emb = self.embeddings_cache.get(
                    key=text, collection="embeddings"
                )
                if cached_emb is not None:
                    cached_key = next(iter(cached_emb.keys()))
                    text_embedding = cached_emb[cached_key]
                else:
                    if self.rate_limiter is not None:
                        self.rate_limiter.acquire()
                    text_embedding = self._get_text_embedding(text)
                    self.embeddings_cache.put(
                        key=text,
                        val={str(uuid.uuid4()): text_embedding},
                        collection="embeddings",
                    )

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    async def aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    await self.rate_limiter.async_acquire()
                text_embedding = await self._aget_text_embedding(text)
            elif self.embeddings_cache is not None:
                cached_emb = await self.embeddings_cache.aget(
                    key=text, collection="embeddings"
                )
                if cached_emb is not None:
                    cached_key = next(iter(cached_emb.keys()))
                    text_embedding = cached_emb[cached_key]
                else:
                    if self.rate_limiter is not None:
                        await self.rate_limiter.async_acquire()
                    text_embedding = await self._aget_text_embedding(text)
                    await self.embeddings_cache.aput(
                        key=text,
                        val={str(uuid.uuid4()): text_embedding},
                        collection="embeddings",
                    )

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatcher.event(
            EmbeddingEndEvent(
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
    ) -> List[Embedding]:
        """Get a list of text embeddings, with batching."""
        cur_batch: List[str] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    EmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    if self.rate_limiter is not None:
                        self.rate_limiter.acquire()
                    if not self.embeddings_cache:
                        embeddings = self._get_text_embeddings(cur_batch)
                    elif self.embeddings_cache is not None:
                        embeddings = self._get_text_embeddings_cached(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                dispatcher.event(
                    EmbeddingEndEvent(
                        chunks=cur_batch,
                        embeddings=embeddings,
                    )
                )
                cur_batch = []

        return result_embeddings

    @dispatcher.span
    async def aget_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[Embedding]:
        """Asynchronously get a list of text embeddings, with batching."""
        num_workers = self.num_workers

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        cur_batch: List[str] = []
        embeddings_coroutines: List[Coroutine] = []
        callback_payloads: List[Tuple[str, List[str]]] = []

        # for idx, text in queue_with_progress:
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    EmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))

                if not self.embeddings_cache:
                    embeddings_coroutines.append(
                        self._aget_text_embeddings_rate_limited(cur_batch)
                    )
                elif self.embeddings_cache is not None:
                    embeddings_coroutines.append(
                        self._aget_text_embeddings_cached(cur_batch)
                    )

                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        if len(embeddings_coroutines) > 0:
            if num_workers and num_workers > 1:
                nested_embeddings = await run_jobs(
                    embeddings_coroutines,
                    show_progress=show_progress,
                    workers=self.num_workers,
                    desc="Generating embeddings",
                )
            elif show_progress:
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
        else:
            nested_embeddings = []

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, text_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            dispatcher.event(
                EmbeddingEndEvent(
                    chunks=text_batch,
                    embeddings=embeddings,
                )
            )
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: text_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Get embedding similarity."""
        return similarity(embedding1=embedding1, embedding2=embedding2, mode=mode)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        show_progress = kwargs.pop("show_progress", False)
        mixed_indices: List[int] = []
        text_indices: List[int] = []
        for i, node in enumerate(nodes):
            mixed_content = node.get_mixed_embedding_content(
                metadata_mode=MetadataMode.EMBED
            )
            if mixed_content is not None and self.supports_mixed_embedding:
                mixed_indices.append(i)
            else:
                text_indices.append(i)

        # Batch text embeddings
        text_nodes = [nodes[i] for i in text_indices]
        if text_nodes:
            text_embeddings = self.get_text_embedding_batch(
                [
                    node.get_content(metadata_mode=MetadataMode.EMBED)
                    for node in text_nodes
                ],
                show_progress=show_progress,
                **kwargs,
            )
        else:
            text_embeddings = []

        # Mixed embeddings (batched)
        mixed_contents: List[MixedEmbeddingContent] = []
        for i in mixed_indices:
            content = nodes[i].get_mixed_embedding_content(
                metadata_mode=MetadataMode.EMBED
            )
            if content is not None:
                mixed_contents.append(content)
        if mixed_contents:
            mixed_embeddings = self.embed_batch(
                mixed_contents,
                show_progress=show_progress,
                **kwargs,
            )
        else:
            mixed_embeddings = []

        # Assign in original order
        text_pos = 0
        mixed_pos = 0
        for i, node in enumerate(nodes):
            if i in mixed_indices:
                node.embedding = mixed_embeddings[mixed_pos]
                mixed_pos += 1
            else:
                node.embedding = text_embeddings[text_pos]
                text_pos += 1

        return nodes

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        show_progress = kwargs.pop("show_progress", False)
        mixed_indices = []
        text_indices = []
        for i, node in enumerate(nodes):
            mixed_content = node.get_mixed_embedding_content(
                metadata_mode=MetadataMode.EMBED
            )
            if mixed_content is not None and self.supports_mixed_embedding:
                mixed_indices.append(i)
            else:
                text_indices.append(i)

        text_nodes = [nodes[i] for i in text_indices]
        if text_nodes:
            text_embeddings = await self.aget_text_embedding_batch(
                [
                    node.get_content(metadata_mode=MetadataMode.EMBED)
                    for node in text_nodes
                ],
                show_progress=show_progress,
                **kwargs,
            )
        else:
            text_embeddings = []

        mixed_contents = []
        for i in mixed_indices:
            content = nodes[i].get_mixed_embedding_content(
                metadata_mode=MetadataMode.EMBED
            )
            if content is not None:
                mixed_contents.append(content)
        if mixed_contents:
            mixed_embeddings = await self.aembed_batch(
                mixed_contents,
                show_progress=show_progress,
                **kwargs,
            )
        else:
            mixed_embeddings = []

        text_pos = 0
        mixed_pos = 0
        for i, node in enumerate(nodes):
            if i in mixed_indices:
                node.embedding = mixed_embeddings[mixed_pos]
                mixed_pos += 1
            else:
                node.embedding = text_embeddings[text_pos]
                text_pos += 1

        return nodes
