"""Base retriever."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    QueryType,
    TextNode,
)
from llama_index.core.settings import Settings
from llama_index.core.utils import print_text
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class BaseRetriever(PromptMixin, DispatcherSpanMixin):
    """Base retriever."""

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[Dict] = None,
        objects: Optional[List[IndexNode]] = None,
        verbose: bool = False,
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager()

        if objects is not None:
            object_map = {obj.index_id: obj.obj for obj in objects}

        self.object_map = object_map or {}
        self._verbose = verbose

    def _check_callback_manager(self) -> None:
        """Check callback manager."""
        if not hasattr(self, "callback_manager"):
            self.callback_manager = Settings.callback_manager

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def _retrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
    ) -> List[NodeWithScore]:
        """Retrieve nodes from object."""
        if self._verbose:
            print_text(
                f"Retrieving from object {obj.__class__.__name__} with query {query_bundle.query_str}\n",
                color="llama_pink",
            )
        if isinstance(obj, NodeWithScore):
            return [obj]
        elif isinstance(obj, BaseNode):
            return [NodeWithScore(node=obj, score=score)]
        elif isinstance(obj, BaseQueryEngine):
            response = obj.query(query_bundle)
            return [
                NodeWithScore(
                    node=TextNode(text=str(response), metadata=response.metadata or {}),
                    score=score,
                )
            ]
        elif isinstance(obj, BaseRetriever):
            return obj.retrieve(query_bundle)
        else:
            raise ValueError(f"Object {obj} is not retrievable.")

    async def _aretrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
    ) -> List[NodeWithScore]:
        """Retrieve nodes from object."""
        if isinstance(obj, NodeWithScore):
            return [obj]
        elif isinstance(obj, BaseNode):
            return [NodeWithScore(node=obj, score=score)]
        elif isinstance(obj, BaseQueryEngine):
            response = await obj.aquery(query_bundle)
            return [NodeWithScore(node=TextNode(text=str(response)), score=score)]
        elif isinstance(obj, BaseRetriever):
            return await obj.aretrieve(query_bundle)
        else:
            raise ValueError(f"Object {obj} is not retrievable.")

    def _handle_recursive_retrieval(
        self, query_bundle: QueryBundle, nodes: List[NodeWithScore]
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
                            obj, query_bundle=query_bundle, score=score
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
        self, query_bundle: QueryBundle, nodes: List[NodeWithScore]
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
                            obj, query_bundle=query_bundle, score=score
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
            if not (
                (n.node.hash, n.node.ref_doc_id) in seen
                or seen.add((n.node.hash, n.node.ref_doc_id))  # type: ignore[func-returns-value]
            )
        ]

    @dispatcher.span
    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """
        Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        self._check_callback_manager()
        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                nodes = self._handle_recursive_retrieval(query_bundle, nodes)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    @dispatcher.span
    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        self._check_callback_manager()

        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle=query_bundle)
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes given query.

        Implemented by the user.

        """

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)
