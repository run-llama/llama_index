import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Collection,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    overload,
)
from typing_extensions import LiteralString, assert_never

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings.omni_modal_base import (
    KD,
    KQ,
    Modalities,
    OmniModalEmbeddingBundle,
)
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.schema import (
    IndexNode,
    NodeWithScore,
    ObjectType,
    QueryBundle,
    QueryType,
)
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
)
from llama_index.core.utils import print_text
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
import llama_index.core.instrumentation as instrument

if TYPE_CHECKING:
    from llama_index.core.indices.omni_modal.base import OmniModalVectorStoreIndex

dispatcher = instrument.get_dispatcher(__name__)


_KD = TypeVar("_KD", bound=LiteralString)
_KQ = TypeVar("_KQ", bound=LiteralString)


class OmniModalVectorIndexRetriever(BaseRetriever, Generic[KD, KQ]):
    """Omni-Modal Vector index retriever.

    Args:
        index (OmniModalVectorStoreIndex): Omni Modal vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    def __init__(
        self,
        index: "OmniModalVectorStoreIndex[KD, KQ]",
        # The similarity_top_k to use for each document modality, defaulting to DEFAULT_SIMILARITY_TOP_K
        similarity_top_k: Optional[Mapping[KD, int]] = None,
        # The sparse_top_k to use for each document modality, defaulting to None
        sparse_top_k: Optional[Mapping[KD, int]] = None,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if similarity_top_k is None:
            similarity_top_k = {}
        if sparse_top_k is None:
            sparse_top_k = {}

        self._index = index

        self._vector_stores = self._index.vector_stores
        self._embed_model = self._index.embed_model

        self._similarity_top_k = {
            modality_key: similarity_top_k.get(modality_key, DEFAULT_SIMILARITY_TOP_K)
            for modality_key in self._embed_model.document_modalities
        }
        self._sparse_top_k = {
            modality_key: sparse_top_k.get(modality_key, None)
            for modality_key in self._embed_model.document_modalities
        }

        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters

        self._kwargs: Mapping[str, Any] = kwargs.get("vector_store_kwargs", {})
        self.callback_manager = (
            callback_manager
            or callback_manager_from_settings_or_context(
                Settings, self._service_context
            )
        )

    @property
    def embed_model(self) -> OmniModalEmbeddingBundle[KD, KQ]:
        return self._embed_model

    @property
    def similarity_top_k(self) -> Mapping[KD, int]:
        return self._similarity_top_k

    @property
    def sparse_top_k(self) -> Mapping[KD, Optional[int]]:
        return self._sparse_top_k

    def _build_vector_store_query(
        self,
        query_bundle_with_embeddings: QueryBundle,
        *,
        similarity_top_k: int,
        sparse_top_k: Optional[int],
    ) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=similarity_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=sparse_top_k,
        )

    def _get_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        vector_store: BasePydanticVectorStore,
        *,
        similarity_top_k: int,
        sparse_top_k: Optional[int],
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(
            query_bundle_with_embeddings,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
        )
        query_result = vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(vector_store, query_result)

    async def _aget_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        vector_store: BasePydanticVectorStore,
        *,
        similarity_top_k: int,
        sparse_top_k: Optional[int],
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(
            query_bundle_with_embeddings,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
        )
        query_result = await vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(vector_store, query_result)

    def _build_node_list_from_query_result(
        self,
        vector_store: BasePydanticVectorStore,
        query_result: VectorStoreQueryResult,
    ) -> List[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                source_node = query_result.nodes[i].source_node
                if (not vector_store.stores_text) or (
                    source_node is not None and source_node.node_type != ObjectType.TEXT
                ):
                    node_id = query_result.nodes[i].node_id
                    if self._docstore.document_exists(node_id):
                        query_result.nodes[i] = self._docstore.get_node(node_id)  # type: ignore[index]

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores

    def _retrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
        *,
        query_type: KQ,
    ) -> List[NodeWithScore]:
        """Retrieve nodes from object."""
        if isinstance(obj, type(self)):
            if self._verbose:
                print_text(
                    f"Retrieving from object {obj.__class__.__name__} with query {query_bundle.query_str}\n",
                    color="llama_pink",
                )

                return obj.retrieve_multi_modal(query_bundle, query_type=query_type)

        return super()._retrieve_from_object(obj, query_bundle, score)

    async def _aretrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
        *,
        query_type: KQ,
    ) -> List[NodeWithScore]:
        """Retrieve nodes from object."""
        if isinstance(obj, type(self)):
            if self._verbose:
                print_text(
                    f"Retrieving from object {obj.__class__.__name__} with query {query_bundle.query_str}\n",
                    color="llama_pink",
                )

                return await obj.aretrieve_multi_modal(
                    query_bundle, query_type=query_type
                )

        return await super()._aretrieve_from_object(obj, query_bundle, score)

    def _handle_recursive_retrieval(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        *,
        query_type: KQ,
    ) -> List[NodeWithScore]:
        retrieved_nodes: List[NodeWithScore] = []
        for n in nodes:
            node = n.node
            score = n.score or 1.0
            if isinstance(node, IndexNode):
                obj = node.obj or self.object_map.get(node.index_id, None)
                if obj is not None:
                    if self._verbose:
                        print_text(
                            f"Retrieval entering {node.index_id}: {obj.__class__.__name__}\n",
                            color="llama_turquoise",
                        )
                    retrieved_nodes.extend(
                        self._retrieve_from_object(
                            obj,
                            query_bundle=query_bundle,
                            score=score,
                            query_type=query_type,
                        )
                    )
                else:
                    retrieved_nodes.append(n)
            else:
                retrieved_nodes.append(n)

        seen = set()
        return [
            n
            for n in retrieved_nodes
            if not (n.node.hash in seen or seen.add(n.node.hash))  # type: ignore[func-returns-value]
        ]

    async def _ahandle_recursive_retrieval(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        *,
        query_type: KQ,
    ) -> List[NodeWithScore]:
        retrieved_nodes: List[NodeWithScore] = []
        for n in nodes:
            node = n.node
            score = n.score or 1.0
            if isinstance(node, IndexNode):
                obj = node.obj or self.object_map.get(node.index_id, None)
                if obj is not None:
                    if self._verbose:
                        print_text(
                            f"Retrieval entering {node.index_id}: {obj.__class__.__name__}\n",
                            color="llama_turquoise",
                        )
                    # TODO: Add concurrent execution via `run_jobs()` ?
                    retrieved_nodes.extend(
                        await self._aretrieve_from_object(
                            obj,
                            query_bundle=query_bundle,
                            score=score,
                            query_type=query_type,
                        )
                    )
                else:
                    retrieved_nodes.append(n)
            else:
                retrieved_nodes.append(n)

        # remove any duplicates based on hash and ref_doc_id
        seen = set()
        return [
            n
            for n in retrieved_nodes
            if not ((n.node.hash, n.node.ref_doc_id) in seen or seen.add((n.node.hash, n.node.ref_doc_id)))  # type: ignore[func-returns-value]
        ]

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        *,
        query_type: KQ,
        doc_types: Collection[KD],
    ) -> List[NodeWithScore]:
        res = []

        embedders = self._embed_model.get_query_document_embed_models(query_type)

        for doc_type in doc_types:
            vector_store = self._vector_stores[doc_type]
            similarity_top_k = self._similarity_top_k[doc_type]
            sparse_top_k = self._sparse_top_k[doc_type]
            embedder = embedders[doc_type]

            if vector_store.is_embedding_query:
                query_bundle.embedding = embedder.embed_query(query_type, query_bundle)
            else:
                query_bundle.embedding = None

            res.extend(
                self._get_nodes_with_embeddings(
                    query_bundle,
                    vector_store,
                    similarity_top_k=similarity_top_k,
                    sparse_top_k=sparse_top_k,
                )
            )

        return res

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
        *,
        query_type: KQ,
        doc_types: Collection[KD],
    ) -> List[NodeWithScore]:
        # Run the two retrievals in async, and return their results as a concatenated list
        results: List[NodeWithScore] = []
        tasks: List[Awaitable[List[NodeWithScore]]] = []

        embedders = self._embed_model.get_query_document_embed_models(query_type)

        for doc_type in doc_types:
            vector_store = self._vector_stores[doc_type]
            similarity_top_k = self._similarity_top_k[doc_type]
            sparse_top_k = self._sparse_top_k[doc_type]
            embedder = embedders[doc_type]

            if vector_store.is_embedding_query:
                query_bundle.embedding = await embedder.aembed_query(
                    query_type, query_bundle
                )
            else:
                query_bundle.embedding = None

            tasks.append(
                self._aget_nodes_with_embeddings(
                    query_bundle,
                    vector_store,
                    similarity_top_k=similarity_top_k,
                    sparse_top_k=sparse_top_k,
                )
            )

        task_results = await asyncio.gather(*tasks)

        for task_result in task_results:
            results.extend(task_result)

        return results

    @dispatcher.span
    def retrieve_multi_modal(
        self,
        query_bundle: QueryBundle,
        *,
        query_type: KQ,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> List[NodeWithScore]:
        if doc_types is None:
            doc_types = self._embed_model.document_modalities.keys()

        self._check_callback_manager()
        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=query_bundle,
            )
        )
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(
                    query_bundle, query_type=query_type, doc_types=doc_types
                )
                nodes = self._handle_recursive_retrieval(
                    query_bundle, nodes, query_type=query_type
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    @dispatcher.span
    async def aretrieve_multi_modal(
        self,
        query_bundle: QueryBundle,
        *,
        query_type: KQ,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> List[NodeWithScore]:
        if doc_types is None:
            doc_types = self._embed_model.document_modalities.keys()

        self._check_callback_manager()
        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=query_bundle,
            )
        )
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(
                    query_bundle, query_type=query_type, doc_types=doc_types
                )
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle, nodes, query_type=query_type
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    def _as_query_bundle(
        self, str_or_query_bundle: QueryType, query_type: Literal["text", "image"]
    ) -> QueryBundle:
        if isinstance(str_or_query_bundle, str):
            if query_type == "text":
                return QueryBundle(str_or_query_bundle)
            elif query_type == "image":
                return QueryBundle("", image_path=str_or_query_bundle)

            assert_never(query_type)

        return str_or_query_bundle

    @overload
    def _as_bimodal(
        self,
        *,
        query_type: _KQ,
        doc_type: _KD,
    ) -> "OmniModalVectorIndexRetriever[_KD, _KQ]":
        ...

    @overload
    def _as_bimodal(
        self,
        *,
        query_type: _KQ,
        doc_type: None = None,
    ) -> "OmniModalVectorIndexRetriever[KD, _KQ]":
        ...

    @overload
    def _as_bimodal(
        self,
        *,
        query_type: None = None,
        doc_type: _KD,
    ) -> "OmniModalVectorIndexRetriever[_KD, KQ]":
        ...

    def _as_bimodal(
        self,
        *,
        query_type: Optional[_KQ] = None,
        doc_type: Optional[_KD] = None,
    ) -> "OmniModalVectorIndexRetriever[Any, Any]":
        """Perform a checked cast of the modality types supported by this retriever."""
        embed_model: OmniModalEmbeddingBundle[Any, Any] = self._embed_model

        try:
            if query_type is not None:
                embed_model.get_query_embed_models(query_type)

            if doc_type is not None:
                embed_model.get_document_embed_model(doc_type)
        except Exception:
            raise

        return self

    def retrieve(
        self,
        str_or_query_bundle: QueryType,
        *,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> List[NodeWithScore]:
        query_bundle = self._as_query_bundle(
            str_or_query_bundle, query_type=Modalities.TEXT.key
        )

        retriever = self._as_bimodal(query_type=Modalities.TEXT.key)

        return retriever.retrieve_multi_modal(
            query_bundle, query_type=Modalities.TEXT.key, doc_types=doc_types
        )

    async def aretrieve(
        self,
        str_or_query_bundle: QueryType,
        *,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> List[NodeWithScore]:
        query_bundle = self._as_query_bundle(
            str_or_query_bundle, query_type=Modalities.TEXT.key
        )

        retriever = self._as_bimodal(query_type=Modalities.TEXT.key)

        return await retriever.aretrieve_multi_modal(
            query_bundle, query_type=Modalities.TEXT.key, doc_types=doc_types
        )

    # Compatibility methods to maintain the same interface as MultiModalRetriever
    # These methods may be removed in the future

    def _retrieve_bimodal(
        self,
        str_or_query_bundle: QueryType,
        query_type: Literal["text", "image"],
        doc_type: Literal["text", "image"],
    ) -> List[NodeWithScore]:
        query_bundle = self._as_query_bundle(str_or_query_bundle, query_type=query_type)

        retriever = self._as_bimodal(query_type=query_type, doc_type=doc_type)

        return retriever.retrieve_multi_modal(
            query_bundle, query_type=query_type, doc_types={doc_type}
        )

    async def _aretrieve_bimodal(
        self,
        str_or_query_bundle: QueryType,
        query_type: Literal["text", "image"],
        doc_type: Literal["text", "image"],
    ) -> List[NodeWithScore]:
        query_bundle = self._as_query_bundle(str_or_query_bundle, query_type=query_type)

        retriever = self._as_bimodal(query_type=query_type, doc_type=doc_type)

        return await retriever.aretrieve_multi_modal(
            query_bundle, query_type=query_type, doc_types={doc_type}
        )

    def text_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        return self._retrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.TEXT.key,
            doc_type=Modalities.TEXT.key,
        )

    async def atext_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        return await self._aretrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.TEXT.key,
            doc_type=Modalities.TEXT.key,
        )

    def text_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        return self._retrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.TEXT.key,
            doc_type=Modalities.IMAGE.key,
        )

    async def atext_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        return await self._aretrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.TEXT.key,
            doc_type=Modalities.IMAGE.key,
        )

    def image_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        return self._retrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.IMAGE.key,
            doc_type=Modalities.IMAGE.key,
        )

    async def aimage_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        return await self._aretrieve_bimodal(
            str_or_query_bundle,
            query_type=Modalities.IMAGE.key,
            doc_type=Modalities.IMAGE.key,
        )
