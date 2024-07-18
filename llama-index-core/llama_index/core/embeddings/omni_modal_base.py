"""Base embeddings file."""

import asyncio
import json
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import cached_property
from io import BytesIO
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterator,
    ItemsView,
    List,
    Literal,
    Optional,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import LiteralString, Self, assert_never

from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
    SimilarityMode,
    mean_agg,
    similarity,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    ImageType,
    MetadataMode,
    QueryBundle,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable, find_duplicates
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
import llama_index.core.instrumentation as instrument

from .multi_modal_base import MultiModalEmbedding

dispatcher = instrument.get_dispatcher(__name__)


K = TypeVar("K", bound=str)
K_co = TypeVar("K_co", bound=str, covariant=True)

N = TypeVar("N", bound=BaseNode)

D_co = TypeVar("D_co", covariant=True)


@dataclass(frozen=True)
class NodeProcessor(Generic[N, D_co]):
    node_type: Type[N]
    """The type of node to process."""

    data_extractor: Callable[[N], D_co]
    """Extracts the data from a node so that it can be passed to :class:`OmniModalEmbedding`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(node_type={self.node_type.__qualname__})"


P = TypeVar("P", bound="NodeProcessor")


class NodeProcessors:
    """Node processors used by the core library."""

    TEXT = NodeProcessor(
        node_type=TextNode,
        data_extractor=lambda node: node.get_content(metadata_mode=MetadataMode.EMBED),
    )
    IMAGE = NodeProcessor(
        node_type=ImageNode,
        data_extractor=lambda node: node.resolve_image(),
    )

    @classmethod
    def group_nodes(
        cls,
        nodes: Collection[N],
        processors: Collection[P],
    ) -> ItemsView[P, List[N]]:
        processor_by_node_type: Dict[Type[N], P] = {}
        nodes_by_processor: Dict[P, List[N]] = defaultdict(list)

        for node in nodes:
            node_type = type(node)
            matched_processor = processor_by_node_type.get(node_type)

            if matched_processor is None:
                node_mro = node_type.mro()
                processor_by_mro_idx = {
                    node_mro.index(processor.node_type): processor
                    for processor in processors
                    if isinstance(node, processor.node_type)
                }

                if not processor_by_mro_idx:
                    raise ValueError(
                        f"Cannot find compatible processor for node (type: {node_type}). "
                        f"Available processors: {list(processors)}"
                    )

                matched_processor = processor_by_mro_idx[min(processor_by_mro_idx)]
                processor_by_node_type[node_type] = matched_processor

            nodes_by_processor[matched_processor].append(node)

        return nodes_by_processor.items()

    @classmethod
    def group_node_datas(
        cls,
        nodes: Collection[N],
        processors: Collection[P],
    ) -> ItemsView[P, List[Tuple[N, object]]]:
        nodes_by_processor = cls.group_nodes(nodes, processors)

        return {
            processor: [(node, processor.data_extractor(node)) for node in nodes]
            for processor, nodes in nodes_by_processor
        }.items()


@dataclass(frozen=True)
class QueryProcessor(Generic[D_co]):
    data_extractor: Callable[[QueryBundle], Collection[D_co]]
    """Extracts the data from a query so that it can be passed to :class:`OmniModalEmbedding`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class QueryProcessors:
    """Query processors used by the core library."""

    TEXT = QueryProcessor(
        data_extractor=lambda query: query.embedding_strs,
    )
    IMAGE = QueryProcessor(
        data_extractor=lambda query: query.embedding_image,
    )


_K = TypeVar("_K", bound=LiteralString)
_N = TypeVar("_N", bound=BaseNode)
_D = TypeVar("_D")


@dataclass(frozen=True)
class Modality(Generic[K_co, N, D_co]):
    key: K_co
    """The key of the corresponding vector store in `MultiModalVectorStoreIndex.storage_context`."""

    node_processor: NodeProcessor[N, D_co] = field(repr=False)
    """Defines how to process nodes belonging to this modality."""

    query_processor: QueryProcessor[D_co] = field(repr=False)
    """Defines how to process queries belonging to this modality."""

    @staticmethod
    def const(
        *,
        key: _K,
        node_processor: NodeProcessor[_N, _D],
        query_processor: QueryProcessor[_D],
    ) -> "Modality[_K, _N, _D]":
        """Enables a literal value to be assigned to the type parameter ``K``."""
        return Modality(
            key=key,
            node_processor=node_processor,
            query_processor=query_processor,
        )

    def __hash__(self) -> int:
        return hash(self.key)


M = TypeVar("M", bound="Modality")


class Modalities:
    """Modalities used by the core library."""

    TEXT = Modality.const(
        key="text",
        node_processor=NodeProcessors.TEXT,
        query_processor=QueryProcessors.TEXT,
    )
    IMAGE = Modality.const(
        key="image",
        node_processor=NodeProcessors.IMAGE,
        query_processor=QueryProcessors.IMAGE,
    )

    @classmethod
    def group_nodes(
        cls,
        nodes: Collection[N],
        modalities: Collection[M],
    ) -> ItemsView[M, List[N]]:
        processor_to_modality = {
            modality.node_processor: modality for modality in modalities
        }
        processors = processor_to_modality.keys()
        nodes_by_processor = NodeProcessors.group_nodes(nodes, processors)

        return {
            processor_to_modality[processor]: nodes
            for processor, nodes in nodes_by_processor
        }.items()

    @classmethod
    def group_node_datas(
        cls,
        nodes: Collection[N],
        modalities: Collection[M],
    ) -> ItemsView[M, List[Tuple[N, object]]]:
        processor_to_modality = {
            modality.node_processor: modality for modality in modalities
        }
        processors = processor_to_modality.keys()
        node_datas_by_processor = NodeProcessors.group_node_datas(nodes, processors)

        return {
            processor_to_modality[processor]: node_datas
            for processor, node_datas in node_datas_by_processor
        }.items()


_K1 = TypeVar("_K1", bound=str)
_K2 = TypeVar("_K2", bound=str)
_K3 = TypeVar("_K3", bound=str)
_K4 = TypeVar("_K4", bound=str)
_K5 = TypeVar("_K5", bound=str)
_T = TypeVar("_T")


class ModalityBundle(Mapping[K, Modality[K, Any, object]]):
    """Container of :class:`Modality` instances."""

    # For better type annotations
    @overload
    @staticmethod
    def of(
        __m1: Modality[_K1, Any, object],
    ) -> "ModalityBundle[_K1]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: Modality[_K1, Any, object],
        __m2: Modality[_K2, Any, object],
    ) -> "ModalityBundle[Union[_K1, _K2]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: Modality[_K1, Any, object],
        __m2: Modality[_K2, Any, object],
        __m3: Modality[_K3, Any, object],
    ) -> "ModalityBundle[Union[_K1, _K2, _K3]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: Modality[_K1, Any, object],
        __m2: Modality[_K2, Any, object],
        __m3: Modality[_K3, Any, object],
        __m4: Modality[_K4, Any, object],
    ) -> "ModalityBundle[Union[_K1, _K2, _K3, _K4]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: Modality[_K1, Any, object],
        __m2: Modality[_K2, Any, object],
        __m3: Modality[_K3, Any, object],
        __m4: Modality[_K4, Any, object],
        __m5: Modality[_K5, Any, object],
    ) -> "ModalityBundle[Union[_K1, _K2, _K3, _K4, _K5]]":
        ...

    @overload
    @staticmethod
    def of(*modalities: Modality[K_co, Any, object]) -> "ModalityBundle[K_co]":
        ...

    @staticmethod
    def of(*modalities: Modality[K_co, Any, object]) -> "ModalityBundle[K_co]":  # type: ignore
        return ModalityBundle(*modalities)

    def __init__(self, *modalities: Modality[K, Any, object]) -> None:
        super().__init__()

        if duplicate_keys := find_duplicates(modality.key for modality in modalities):
            raise ValueError(f"Found duplicate modality keys: {duplicate_keys}")

        self._modalities_by_key = {modality.key: modality for modality in modalities}

    def __iter__(self) -> Iterator[K]:
        return iter(self._modalities_by_key)

    def __contains__(self, x: object, /) -> bool:
        return x in self._modalities_by_key

    def __len__(self) -> int:
        return len(self._modalities_by_key)

    def __getitem__(self, key: K, /) -> Modality[K, Any, object]:
        return self._modalities_by_key[key]

    @overload
    def get(self, key: K, /) -> Optional[Modality[K, Any, object]]:
        ...

    @overload
    def get(
        self, key: K, /, default: Union[Modality[K, Any, object], _T]
    ) -> Union[Modality[K, Any, object], _T]:
        ...

    def get(
        self, key: K, /, default: Optional[Union[Modality[K, Any, object], _T]] = None
    ) -> Optional[Union[Modality[K, Any, object], _T]]:
        return self._modalities_by_key.get(key, default)

    def items(self):
        return self._modalities_by_key.items()

    def keys(self):
        return self._modalities_by_key.keys()

    def values(self):
        return self._modalities_by_key.values()

    def __eq__(self, other: object, /) -> bool:
        return (
            isinstance(other, ModalityBundle)
            and other._modalities_by_key == self._modalities_by_key
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._modalities_by_key})"


KD = TypeVar("KD", bound=str)
KQ = TypeVar("KQ", bound=str)

TextModality = Modality[Literal["text"], TextNode, str]
ImageModality = Modality[Literal["image"], ImageNode, ImageType]
TextOrImageModality = Union[TextModality, ImageModality]


# Mixing in Generic to existing TransformComponent requires Pydantic V2
@dataclass
class GenericTransformComponent:
    """Base class for transform components."""

    class Config:
        @staticmethod
        def schema_extra(
            schema: Dict[str, Any], model: "GenericTransformComponent"
        ) -> None:
            """Add class name to schema."""
            schema["properties"]["class_name"] = {
                "title": "Class Name",
                "type": "string",
                "default": model.class_name(),
            }

        arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = asdict(self, **kwargs)
        data["class_name"] = self.class_name()
        return data

    def __getstate__(self) -> Dict[str, Any]:
        state = {"__dict__": self.__dict__}

        # tiktoken is not pickleable
        # state["__dict__"] = self.dict()
        state["__dict__"].pop("tokenizer", None)

        # remove local functions
        keys_to_remove = []
        for key, val in state["__dict__"].items():
            if key.endswith("_fn"):
                keys_to_remove.append(key)
            if "<lambda>" in str(val):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            state["__dict__"].pop(key, None)

        # remove private attributes -- kind of dangerous
        state["__private_attribute_values__"] = {}

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Use the __dict__ and __init__ method to set state
        # so that all variable initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            for k, v in state["__dict__"].items():
                setattr(self, k, v)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)

    # Define post-init logic in `model_post_init` to avoid having to rename
    # the method when switching to Pydantic V2
    def __post_init__(self) -> None:
        return self.model_post_init()

    def model_post_init(self) -> None:
        pass


@dataclass
class OmniModalEmbedding(
    GenericTransformComponent, DispatcherSpanMixin, Generic[KD, KQ]
):
    model_name: str = field(default="unknown")
    """The name of the embedding model."""

    embed_batch_size: int = field(default=DEFAULT_EMBED_BATCH_SIZE)
    """The batch size for embedding calls."""

    num_workers: Optional[int] = field(default=None)
    """The number of workers to use for async embedding calls."""

    callback_manager: CallbackManager = field(default_factory=CallbackManager)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().dict(**kwargs)

        # Avoid having to repeat this code like in BaseEmbedding
        data.pop("api_key", None)

        # exclude not supported for dataclass field
        data.pop("callback_manager", None)

        return data

    def model_post_init(self) -> None:
        super().model_post_init()

        # gt/lte not supported for dataclass field
        embed_batch_size = self.embed_batch_size
        if not 0 < embed_batch_size <= 2048:
            msg = f"embed_batch_size is not in the range (0, 2048]. Found: {embed_batch_size}"
            raise ValueError(msg)

    @staticmethod
    def from_base(embed_model: BaseEmbedding):
        return TextToTextEmbedding(
            embed_model=embed_model,
            model_name=embed_model.model_name,
            embed_batch_size=embed_model.embed_batch_size,
            callback_manager=embed_model.callback_manager,
            num_workers=embed_model.num_workers,
        )

    @staticmethod
    def from_multi_modal(
        embed_model: MultiModalEmbedding, *, is_image_to_text: bool = False
    ):
        if is_image_to_text:
            return TextImageToImageEmbedding(
                _document_modalities=ModalityBundle.of(Modalities.TEXT),
                model_name=embed_model.model_name,
                embed_batch_size=embed_model.embed_batch_size,
                callback_manager=embed_model.callback_manager,
                num_workers=embed_model.num_workers,
            )

        return TextImageToImageEmbedding(
            _document_modalities=ModalityBundle.of(Modalities.IMAGE),
            model_name=embed_model.model_name,
            embed_batch_size=embed_model.embed_batch_size,
            callback_manager=embed_model.callback_manager,
            num_workers=embed_model.num_workers,
        )

    @property
    @abstractmethod
    def document_modalities(self) -> ModalityBundle[KD]:
        """The supported modalities for document embeddings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def query_modalities(self) -> ModalityBundle[KQ]:
        """The supported modalities for query embeddings."""
        raise NotImplementedError

    def _embedding_end_event(
        self,
        data_items: List[object],
        embeddings: List[Embedding],
    ) -> EmbeddingEndEvent:
        return EmbeddingEndEvent.construct(
            # You can override __str__ for more user-friendly output
            chunks=[str(data) for data in data_items],
            # Very expensive to validate all items in the embedding vector
            embeddings=embeddings,
        )

    @abstractmethod
    def _get_query_embedding(
        self, modality: Modality[KQ, Any, object], data: object
    ) -> Embedding:
        """
        Embed the input query synchronously.

        When embedding a query, depending on the model, a special instruction
        can be prepended to the raw query string. For example, "Represent the
        question for retrieving supporting documents: ". If you're curious,
        other examples of predefined instructions can be found in
        embeddings/huggingface_utils.py.
        """
        raise NotImplementedError

    async def _aget_query_embedding(
        self, modality: Modality[KQ, Any, object], data: object
    ) -> Embedding:
        """Embed the input query asynchronously."""
        return self._get_query_embedding(modality, data)

    def get_query_modality(self, key: KQ) -> Modality[KQ, Any, object]:
        if key not in self.query_modalities:
            raise ValueError(
                f"The query modality (key={key}) is not supported. "
                f"Supported modalities: {set(self.query_modalities.values())}"
            )

        return self.query_modalities[key]

    @dispatcher.span
    def get_query_embedding(self, modality_key: KQ, data: object) -> Embedding:
        """Embed the input query."""
        modality = self.get_query_modality(modality_key)

        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=self.to_dict(),
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            embedding = self._get_query_embedding(modality, data)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [data],
                    EventPayload.EMBEDDINGS: [embedding],
                },
            )
        dispatcher.event(
            self._embedding_end_event(
                data_items=[data],
                embeddings=[embedding],
            )
        )

        return embedding

    @dispatcher.span
    async def aget_query_embedding(self, modality_key: KQ, data: object) -> Embedding:
        """Asynchronously embed the input query."""
        modality = self.get_query_modality(modality_key)

        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=self.to_dict(),
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            embedding = await self._aget_query_embedding(modality, data)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [data],
                    EventPayload.EMBEDDINGS: [embedding],
                },
            )
        dispatcher.event(
            self._embedding_end_event(
                data_items=[data],
                embeddings=[embedding],
            )
        )

        return embedding

    def get_agg_embedding_from_queries(
        self,
        modality_key: KQ,
        data_items: Collection[object],
        agg_fn: Optional[Callable[[List[Embedding]], Embedding]] = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [
            self.get_query_embedding(modality_key, data) for data in data_items
        ]

        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        modality_key: KQ,
        data_items: Collection[object],
        agg_fn: Optional[Callable[[List[Embedding]], Embedding]] = None,
    ) -> Embedding:
        """Asynchronously get aggregated embedding from multiple queries."""
        tasks = (self.aget_query_embedding(modality_key, data) for data in data_items)
        query_embeddings = await asyncio.gather(*tasks)

        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_document_embedding(
        self, modality: Modality[KD, Any, object], data: object
    ) -> Embedding:
        """Embed the input document synchronously.

        When embedding text, depending on the model, a special instruction
        can be prepended to the raw text string. For example, "Represent the
        document for retrieval: ". If you're curious, other examples of
        predefined instructions can be found in embeddings/huggingface_utils.py.
        """
        raise NotImplementedError

    async def _aget_document_embedding(
        self, modality: Modality[KD, Any, object], data: object
    ) -> Embedding:
        """Embed the input document asynchronously."""
        return self._get_document_embedding(modality, data)

    def _get_document_embeddings(
        self, modality: Modality[KD, Any, object], data_items: List[object]
    ) -> List[Embedding]:
        """Embed the input sequence of document synchronously."""
        return [self._get_document_embedding(modality, data) for data in data_items]

    async def _aget_document_embeddings(
        self, modality: Modality[KD, Any, object], data_items: List[object]
    ) -> List[Embedding]:
        """Embed the input sequence of document asynchronously."""
        tasks = (self._aget_document_embedding(modality, data) for data in data_items)
        return await asyncio.gather(*tasks)

    def get_document_modality(self, key: KD) -> Modality[KD, Any, object]:
        if key not in self.document_modalities:
            raise ValueError(
                f"The document modality (key={key}) is not supported. "
                f"Supported modalities: {set(self.document_modalities.values())}"
            )

        return self.document_modalities[key]

    @dispatcher.span
    def get_document_embedding(self, modality_key: KD, data: object) -> Embedding:
        """Embed the input document."""
        modality = self.get_document_modality(modality_key)

        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=self.to_dict(),
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            embedding = self._get_document_embedding(modality, data)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [data],
                    EventPayload.EMBEDDINGS: [embedding],
                },
            )
        dispatcher.event(
            self._embedding_end_event(
                data_items=[data],
                embeddings=[embedding],
            )
        )

        return embedding

    @dispatcher.span
    async def aget_document_embedding(
        self, modality_key: KD, data: object
    ) -> Embedding:
        """Asynchronously embed the input document."""
        modality = self.get_document_modality(modality_key)

        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=self.to_dict(),
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            embedding = await self._aget_document_embedding(modality, data)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [data],
                    EventPayload.EMBEDDINGS: [embedding],
                },
            )
        dispatcher.event(
            self._embedding_end_event(
                data_items=[data],
                embeddings=[embedding],
            )
        )

        return embedding

    @dispatcher.span
    def get_document_embedding_batch(
        self, modality_key: KD, data_items: List[object], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of document embeddings, with batching."""
        modality = self.get_document_modality(modality_key)

        cur_batch: List[object] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(data_items, show_progress, "Generating embeddings")
        )

        for idx, data in queue_with_progress:
            cur_batch.append(data)
            if idx == len(data_items) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    EmbeddingStartEvent(
                        model_dict=self.to_dict(),
                    )
                )
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_document_embeddings(modality, cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                dispatcher.event(
                    self._embedding_end_event(
                        data_items=cur_batch,
                        embeddings=embeddings,
                    )
                )
                cur_batch = []

        return result_embeddings

    @dispatcher.span
    async def aget_document_embedding_batch(
        self, modality_key: KD, data_items: List[object], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of document embeddings, with batching."""
        modality = self.get_document_modality(modality_key)

        cur_batch: List[object] = []
        callback_payloads: List[Tuple[str, List[object]]] = []
        result_embeddings: List[Embedding] = []
        embeddings_coroutines: List[Awaitable[List[Embedding]]] = []

        for idx, data in enumerate(data_items):
            cur_batch.append(data)
            if idx == len(data_items) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    EmbeddingStartEvent(
                        model_dict=self.to_dict(),
                    )
                )
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(
                    self._aget_document_embeddings(modality, cur_batch)
                )
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
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

        for (event_id, data_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            dispatcher.event(
                self._embedding_end_event(
                    data_items=data_batch,
                    embeddings=embeddings,
                )
            )
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: data_batch,
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

    def embed_query(self, modality_key: KQ, query: QueryBundle) -> Embedding:
        """Embed a query bundle."""
        modality = self.get_query_modality(modality_key)
        data = modality.query_processor.data_extractor(query)

        return self.get_agg_embedding_from_queries(modality_key, data)

    async def aembed_query(self, modality_key: KQ, query: QueryBundle) -> Embedding:
        """Asynchronously embed a query bundle."""
        modality = self.get_query_modality(modality_key)
        data = modality.query_processor.data_extractor(query)

        return await self.aget_agg_embedding_from_queries(modality_key, data)

    def __call__(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Transform a list of nodes."""
        return self.as_collection().embed_nodes(nodes, **kwargs)

    async def acall(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Asynchronously transform a list of nodes."""
        return await self.as_collection().aembed_nodes(nodes, **kwargs)

    def as_collection(self) -> "OmniModalEmbeddingBundle[KD, KQ]":
        return OmniModalEmbeddingBundle.of(self)


_KD1 = TypeVar("_KD1", bound=str)
_KD2 = TypeVar("_KD2", bound=str)
_KD3 = TypeVar("_KD3", bound=str)
_KD4 = TypeVar("_KD4", bound=str)
_KD5 = TypeVar("_KD5", bound=str)

_KQ1 = TypeVar("_KQ1", bound=str)
_KQ2 = TypeVar("_KQ2", bound=str)
_KQ3 = TypeVar("_KQ3", bound=str)
_KQ4 = TypeVar("_KQ4", bound=str)
_KQ5 = TypeVar("_KQ5", bound=str)


@dataclass
class OmniModalEmbeddingBundle(
    GenericTransformComponent, Collection[OmniModalEmbedding[KD, KQ]]
):
    """Container of :class:`OmniModalEmbedding` instances."""

    # For better type annotations
    @overload
    @staticmethod
    def of(
        __m1: OmniModalEmbedding[_KD1, _KQ1],
    ) -> "OmniModalEmbeddingBundle[_KD1, _KQ1]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: OmniModalEmbedding[_KD1, _KQ1],
        __m2: OmniModalEmbedding[_KD2, _KQ2],
    ) -> "OmniModalEmbeddingBundle[Union[_KQ1, _KQ2], Union[_KD1, _KD2]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: OmniModalEmbedding[_KD1, _KQ1],
        __m2: OmniModalEmbedding[_KD2, _KQ2],
        __m3: OmniModalEmbedding[_KD3, _KQ3],
    ) -> "OmniModalEmbeddingBundle[Union[_KQ1, _KQ2, _KQ3], Union[_KD1, _KD2, _KD3]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: OmniModalEmbedding[_KD1, _KQ1],
        __m2: OmniModalEmbedding[_KD2, _KQ2],
        __m3: OmniModalEmbedding[_KD3, _KQ3],
        __m4: OmniModalEmbedding[_KD4, _KQ4],
    ) -> "OmniModalEmbeddingBundle[Union[_KQ1, _KQ2, _KQ3, _KQ4], Union[_KD1, _KD2, _KD3, _KD4]]":
        ...

    @overload
    @staticmethod
    def of(
        __m1: OmniModalEmbedding[_KD1, _KQ1],
        __m2: OmniModalEmbedding[_KD2, _KQ2],
        __m3: OmniModalEmbedding[_KD3, _KQ3],
        __m4: OmniModalEmbedding[_KD4, _KQ4],
        __m5: OmniModalEmbedding[_KD5, _KQ5],
    ) -> "OmniModalEmbeddingBundle[Union[_KQ1, _KQ2, _KQ3, _KQ4, _KD5], Union[_KD1, _KD2, _KD3, _KD4, _KD5]]":
        ...

    @overload
    @staticmethod
    def of(
        *embed_models: OmniModalEmbedding[KD, KQ]
    ) -> "OmniModalEmbeddingBundle[KD, KQ]":
        ...

    @staticmethod
    def of(*embed_models: OmniModalEmbedding[KD, KQ]) -> "OmniModalEmbeddingBundle[KD, KQ]":  # type: ignore
        return OmniModalEmbeddingBundle(_embed_models=embed_models)

    _embed_models: Collection[OmniModalEmbedding[KD, KQ]]

    def __iter__(self) -> Iterator[OmniModalEmbedding[KD, KQ]]:
        return iter(self._embed_models)

    def __contains__(self, x: object, /) -> bool:
        return x in self._embed_models

    def __len__(self) -> int:
        return len(self._embed_models)

    def __bool__(self) -> bool:
        return bool(self._embed_models)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OmniModalEmbeddingBundle)
            and other._embed_model_by_document_modality
            == self._embed_model_by_document_modality
            and other._embed_models_by_query_modality
            == self._embed_models_by_query_modality
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._embed_models)})"

    def model_post_init(self) -> None:
        super().model_post_init()

        embed_models = self._embed_models

        # No duplicates
        document_modalities = [
            modality
            for embed_model in embed_models
            for modality in embed_model.document_modalities.values()
        ]
        if duplicate_document_modalities := find_duplicates(document_modalities):
            raise ValueError(
                f"Found duplicate document modalities: {duplicate_document_modalities}"
            )

        self._embed_model_by_document_modality = {
            modality: embed_model
            for embed_model in embed_models
            for modality in embed_model.document_modalities.values()
        }

        # Allow duplicates
        embed_models_by_query_modality: Dict[
            Modality[KQ, Any, object], List[OmniModalEmbedding[KD, KQ]]
        ] = defaultdict(list)
        for embed_model in embed_models:
            for modality in embed_model.query_modalities.values():
                embed_models_by_query_modality[modality].append(embed_model)

        # So that equality checks remain consistent
        for ms in embed_models_by_query_modality.values():
            ms.sort(key=lambda m: m.document_modalities.keys())

        # Avoid defaultdict behaviour
        self._embed_models_by_query_modality = dict(embed_models_by_query_modality)

    @cached_property
    def document_modalities(self) -> ModalityBundle[KD]:
        return ModalityBundle.of(*self._embed_model_by_document_modality)

    @cached_property
    def query_modalities(self) -> ModalityBundle[KQ]:
        return ModalityBundle.of(*self._embed_models_by_query_modality)

    def get_document_embed_model(self, key: KD) -> OmniModalEmbedding[KD, KQ]:
        if key not in self.document_modalities:
            raise ValueError(
                f"The document modality (key={key}) is not supported. "
                f"Supported modalities: {set(self.document_modalities.values())}"
            )

        doc_modality = self.document_modalities[key]
        return self._embed_model_by_document_modality[doc_modality]

    def get_query_embed_models(self, key: KQ) -> Collection[OmniModalEmbedding[KD, KQ]]:
        if key not in self.query_modalities:
            raise ValueError(
                f"The query modality (key={key}) is not supported. "
                f"Supported modalities: {set(self.query_modalities.values())}"
            )

        modality = self.query_modalities[key]
        return self._embed_models_by_query_modality[modality]

    def get_query_document_embed_models(
        self, key: KQ
    ) -> Mapping[KD, OmniModalEmbedding[KD, KQ]]:
        # Document modality keys are guaranteed to be unique
        return {
            modality_key: embed_model
            for embed_model in self.get_query_embed_models(key)
            for modality_key in embed_model.document_modalities
        }

    def group_queries_by_modality(self, nodes: Collection[BaseNode]):
        return Modalities.group_nodes(nodes, self.query_modalities.values())

    def group_documents_by_modality(self, nodes: Collection[BaseNode]):
        return Modalities.group_nodes(nodes, self.document_modalities.values())

    def embed_nodes(
        self,
        nodes: List[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Embed a list of nodes."""
        embedder_by_modality = self._embed_model_by_document_modality
        node_datas_by_modality = Modalities.group_node_datas(
            nodes, embedder_by_modality.keys()
        )

        for modality, node_datas in node_datas_by_modality:
            embedder = embedder_by_modality[modality]
            node_items = [e[0] for e in node_datas]
            data_items = [e[1] for e in node_datas]

            embeddings = embedder.get_document_embedding_batch(
                modality.key, data_items, show_progress=show_progress
            )

            for node, embedding in zip(node_items, embeddings):
                node.embedding = embedding

        return nodes

    async def aembed_nodes(
        self,
        nodes: List[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously embed a list of nodes."""
        embedder_by_modality = self._embed_model_by_document_modality
        node_datas_by_modality = Modalities.group_node_datas(
            nodes, embedder_by_modality.keys()
        )

        for modality, node_datas in node_datas_by_modality:
            embedder = embedder_by_modality[modality]
            node_items = [e[0] for e in node_datas]
            data_items = [e[1] for e in node_datas]

            embeddings = await embedder.aget_document_embedding_batch(
                modality.key, data_items, show_progress=show_progress
            )

            for node, embedding in zip(node_items, embeddings):
                node.embedding = embedding

        return nodes

    def __call__(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Transform a list of nodes."""
        return self.embed_nodes(nodes, **kwargs)

    async def acall(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Asynchronously transform a list of nodes."""
        return await self.aembed_nodes(nodes, **kwargs)


# Adapters
def validate_text_data(data: object, *, prefix: str) -> str:
    if not isinstance(data, str):
        msg = f"The {prefix} data is not a string. Found: {type(data)}"
        raise TypeError(msg)

    return data


def validate_text_data_items(data_items: List[object], *, prefix: str) -> List[str]:
    return [validate_text_data(data, prefix=prefix) for data in data_items]


def validate_image_data(data: object, *, prefix: str) -> ImageType:
    if not isinstance(data, (str, BytesIO)):
        msg = f"The {prefix} data is not a string or buffer. Found: {type(data)}"
        raise TypeError(msg)

    return data


def validate_image_data_items(
    data_items: List[object], *, prefix: str
) -> List[ImageType]:
    return [validate_image_data(data, prefix=prefix) for data in data_items]


@dataclass
class TextToTextEmbedding(OmniModalEmbedding[Literal["text"], Literal["text"]]):
    embed_model: BaseEmbedding = field(
        default_factory=lambda: resolve_embed_model(None)
    )

    @cached_property
    def document_modalities(self) -> ModalityBundle[Literal["text"]]:
        return ModalityBundle.of(Modalities.TEXT)

    @cached_property
    def query_modalities(self) -> ModalityBundle[Literal["text"]]:
        return ModalityBundle.of(Modalities.TEXT)

    def _get_query_embedding(self, modality: TextModality, data: object) -> Embedding:
        if modality.key == "text":
            data = validate_text_data(data, prefix="query")
            return self.embed_model._get_text_embedding(data)

        assert_never(modality)

    async def _aget_query_embedding(
        self, modality: TextModality, data: object
    ) -> Embedding:
        if modality.key == "text":
            data = validate_text_data(data, prefix="query")
            return await self.embed_model._aget_text_embedding(data)

        assert_never(modality)

    def _get_document_embedding(
        self, modality: TextModality, data: object
    ) -> Embedding:
        data = validate_text_data(data, prefix="document")
        return self.embed_model._get_text_embedding(data)

    async def _aget_document_embedding(
        self, modality: TextModality, data: object
    ) -> Embedding:
        data = validate_text_data(data, prefix="document")
        return await self.embed_model._aget_text_embedding(data)

    def _get_document_embeddings(
        self, modality: TextModality, data_items: List[object]
    ) -> List[Embedding]:
        data_items_ = validate_text_data_items(data_items, prefix="document")
        return self.embed_model._get_text_embeddings(data_items_)

    async def _aget_document_embeddings(
        self, modality: TextModality, data_items: List[object]
    ) -> List[Embedding]:
        data_items_ = validate_text_data_items(data_items, prefix="document")
        return await self.embed_model._aget_text_embeddings(data_items_)


def _mm_default_embed_model():
    embed_model = resolve_embed_model("clip:ViT-B/32")
    assert isinstance(embed_model, MultiModalEmbedding)
    return embed_model


def _mm_default_document_modalities():
    raise ValueError("No document modalities were provided")


@dataclass
class TextImageToImageEmbedding(OmniModalEmbedding[KD, Literal["text", "image"]]):
    embed_model: MultiModalEmbedding = field(default_factory=_mm_default_embed_model)
    _document_modalities: ModalityBundle[KD] = field(
        default_factory=_mm_default_document_modalities
    )

    @property
    def document_modalities(self) -> ModalityBundle[KD]:
        return self._document_modalities

    @cached_property
    def query_modalities(self) -> ModalityBundle[Literal["text", "image"]]:
        return ModalityBundle.of(Modalities.TEXT, Modalities.IMAGE)

    def _get_embedding(
        self, modality: TextOrImageModality, data: object, *, prefix: str
    ) -> Embedding:
        if modality.key == "text":
            data = validate_text_data(data, prefix=prefix)
            return self.embed_model._get_text_embedding(data)
        elif modality.key == "image":
            data = validate_image_data(data, prefix=prefix)
            return self.embed_model._get_image_embedding(data)

        assert_never(modality)

    async def _aget_embedding(
        self, modality: TextOrImageModality, data: object, *, prefix: str
    ) -> Embedding:
        if modality.key == "text":
            data = validate_text_data(data, prefix=prefix)
            return await self.embed_model._aget_text_embedding(data)
        elif modality.key == "image":
            data = validate_image_data(data, prefix=prefix)
            return await self.embed_model._aget_image_embedding(data)

        assert_never(modality)

    def _get_embeddings(
        self, modality: TextOrImageModality, data_items: List[object], *, prefix: str
    ) -> List[Embedding]:
        if modality.key == "text":
            data_items_ = validate_text_data_items(data_items, prefix=prefix)
            return self.embed_model._get_text_embeddings(data_items_)
        elif modality.key == "image":
            data_items_ = validate_image_data_items(data_items, prefix=prefix)
            return self.embed_model._get_image_embeddings(data_items_)

        assert_never(modality)

    async def _aget_embeddings(
        self, modality: TextOrImageModality, data_items: List[object], *, prefix: str
    ) -> List[Embedding]:
        if modality.key == "text":
            data_items_ = validate_text_data_items(data_items, prefix=prefix)
            return await self.embed_model._aget_text_embeddings(data_items_)
        elif modality.key == "image":
            data_items_ = validate_image_data_items(data_items, prefix=prefix)
            return await self.embed_model._aget_image_embeddings(data_items_)

        assert_never(modality)

    def _get_query_embedding(
        self, modality: TextOrImageModality, data: object
    ) -> Embedding:
        return self._get_embedding(modality, data, prefix="query")

    async def _aget_query_embedding(
        self, modality: TextOrImageModality, data: object
    ) -> Embedding:
        return await self._aget_embedding(modality, data, prefix="query")

    def _get_document_embedding(
        self, modality: TextOrImageModality, data: object
    ) -> Embedding:
        return self._get_embedding(modality, data, prefix="document")

    async def _aget_document_embedding(
        self, modality: TextOrImageModality, data: object
    ) -> Embedding:
        return await self._aget_embedding(modality, data, prefix="document")

    def _get_document_embeddings(
        self, modality: TextOrImageModality, data_items: List[object]
    ) -> List[Embedding]:
        return self._get_embeddings(modality, data_items, prefix="document")

    async def _aget_document_embeddings(
        self, modality: TextOrImageModality, data_items: List[object]
    ) -> List[Embedding]:
        return await self._aget_embeddings(modality, data_items, prefix="document")
