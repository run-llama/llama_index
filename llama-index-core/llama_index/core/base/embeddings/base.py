"""Base embeddings file."""

import asyncio
import uuid
from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
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

# TODO: change to numpy array
Embedding = List[float]


class EmbeddingResponse:
    """
    Response from an embedding call, carrying the vector and optional metadata.

    Mirrors ``CompletionResponse`` / ``ChatResponse`` for LLMs: integrations
    that can report provider-side token usage return an ``EmbeddingResponse``
    instead of a plain ``List[float]``.  Integrations that return a plain list
    continue to work unchanged.

    Args:
        embedding: The embedding vector.
        token_count: Provider-reported token count for this call, if available.
        raw: The raw API response object, for caller inspection.

    """

    __slots__ = ("embedding", "token_count", "raw")

    def __init__(
        self,
        embedding: Embedding,
        token_count: Optional[int] = None,
        raw: Optional[Any] = None,
    ) -> None:
        self.embedding = embedding
        self.token_count = token_count
        self.raw = raw


EmbeddingResultType = Union[Embedding, EmbeddingResponse]
BatchEmbeddingResultType = List[EmbeddingResultType]


def _unpack_embedding(result: EmbeddingResultType) -> Tuple[Embedding, Optional[int]]:
    """
    Unpack an embedding result into (vector, token_count).

    Accepts both plain ``List[float]`` (backwards-compatible) and the new
    ``EmbeddingResponse``.
    """
    if isinstance(result, EmbeddingResponse):
        return result.embedding, result.token_count
    return result, None


def _unpack_embeddings(
    results: BatchEmbeddingResultType,
) -> Tuple[List[Embedding], Optional[int]]:
    """
    Unpack a batch of embedding results.

    Returns (embeddings, total_token_count).  Token counts are summed across
    all items that report them; the total is ``None`` only when *no* item
    carried a count.
    """
    embeddings: List[Embedding] = []
    total: Optional[int] = None
    for r in results:
        emb, tc = _unpack_embedding(r)
        embeddings.append(emb)
        if tc is not None:
            total = (total or 0) + tc
    return embeddings, total


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
    def _get_query_embedding(self, query: str) -> EmbeddingResultType:
        """
        Embed the input query synchronously.

        Subclasses should implement this method.  May return a plain
        ``List[float]`` (backwards-compatible) or an ``EmbeddingResponse``
        carrying the vector together with provider-reported token usage.
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> EmbeddingResultType:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method.  May return a plain
        ``List[float]`` (backwards-compatible) or an ``EmbeddingResponse``
        carrying the vector together with provider-reported token usage.
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
        token_count: Optional[int] = None
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                query_embedding, token_count = _unpack_embedding(
                    self._get_query_embedding(query)
                )
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
                    query_embedding, token_count = _unpack_embedding(
                        self._get_query_embedding(query)
                    )
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
                token_count=token_count,
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
        token_count: Optional[int] = None
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    await self.rate_limiter.async_acquire()
                query_embedding, token_count = _unpack_embedding(
                    await self._aget_query_embedding(query)
                )
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
                    query_embedding, token_count = _unpack_embedding(
                        await self._aget_query_embedding(query)
                    )
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
                token_count=token_count,
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
    def _get_text_embedding(self, text: str) -> EmbeddingResultType:
        """
        Embed the input text synchronously.

        Subclasses should implement this method.  May return a plain
        ``List[float]`` (backwards-compatible) or an ``EmbeddingResponse``
        carrying the vector together with provider-reported token usage.
        """

    async def _aget_text_embedding(self, text: str) -> EmbeddingResultType:
        """
        Embed the input text asynchronously.

        Subclasses can implement this method if there is a true async
        implementation.
        """
        # Default implementation just falls back on _get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> BatchEmbeddingResultType:
        """
        Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> BatchEmbeddingResultType:
        """
        Embed the input sequence of text asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    async def _aget_text_embeddings_rate_limited(
        self, texts: List[str]
    ) -> BatchEmbeddingResultType:
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
            raw_results = self._get_text_embeddings([x[1] for x in non_cached_texts])
            unpacked, _ = _unpack_embeddings(raw_results)
            for j, text_embedding in enumerate(unpacked):
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
            raw_results = await self._aget_text_embeddings(
                [x[1] for x in non_cached_texts]
            )
            unpacked, _ = _unpack_embeddings(raw_results)
            for j, text_embedding in enumerate(unpacked):
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
        token_count: Optional[int] = None
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                text_embedding, token_count = _unpack_embedding(
                    self._get_text_embedding(text)
                )
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
                    text_embedding, token_count = _unpack_embedding(
                        self._get_text_embedding(text)
                    )
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
                token_count=token_count,
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
        token_count: Optional[int] = None
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            if not self.embeddings_cache:
                if self.rate_limiter is not None:
                    await self.rate_limiter.async_acquire()
                text_embedding, token_count = _unpack_embedding(
                    await self._aget_text_embedding(text)
                )
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
                    text_embedding, token_count = _unpack_embedding(
                        await self._aget_text_embedding(text)
                    )
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
                token_count=token_count,
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
                        embeddings, token_count = _unpack_embeddings(
                            self._get_text_embeddings(cur_batch)
                        )
                    elif self.embeddings_cache is not None:
                        embeddings = self._get_text_embeddings_cached(cur_batch)
                        token_count = None
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
                        token_count=token_count,
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

        result_embeddings: List[Embedding] = []
        unpacked_batches: List[Tuple[List[Embedding], Optional[int]]] = []
        for batch_results in nested_embeddings:
            embeddings, token_count = _unpack_embeddings(batch_results)
            result_embeddings.extend(embeddings)
            unpacked_batches.append((embeddings, token_count))

        for (event_id, text_batch), (embeddings, token_count) in zip(
            callback_payloads, unpacked_batches
        ):
            dispatcher.event(
                EmbeddingEndEvent(
                    chunks=text_batch,
                    embeddings=embeddings,
                    token_count=token_count,
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
        embeddings = self.get_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes
