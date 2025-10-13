import os
from PIL import Image


from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore, ImageDocument
from llama_index.core.llms import ImageBlock
from llama_index.core.base.response.schema import RESPONSE_TYPE
from typing import Optional, List, Union
from typing_extensions import override
from .retriever import LanceDBRetriever, ExtendedQueryBundle

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class LanceDBRetrieverQueryEngine(RetrieverQueryEngine):
    def __init__(
        self,
        retriever: LanceDBRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: List[BaseNodePostprocessor] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            retriever, response_synthesizer, node_postprocessors, callback_manager
        )

    @override
    def retrieve(self, query_bundle: ExtendedQueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever._retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    @override
    async def aretrieve(self, query_bundle: ExtendedQueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever._aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    @override
    @dispatcher.span
    def _query(self, query_bundle: ExtendedQueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @override
    @dispatcher.span
    async def _aquery(self, query_bundle: ExtendedQueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self.aretrieve(query_bundle)

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @override
    @dispatcher.span
    def query(
        self,
        query_str: Optional[str] = None,
        query_image: Optional[
            Union[Image.Image, ImageBlock, ImageDocument, str]
        ] = None,
        query_image_path: Optional[os.PathLike[str]] = None,
    ) -> RESPONSE_TYPE:
        """
        Executes a query against the managed LanceDB index.

        Args:
            query_str (Optional[str]): The text query string to search for. Defaults to None.
            query_image (Optional[Union[Image.Image, ImageBlock, ImageDocument, str]]): An image or image-like object to use as part of the query. Can be a PIL Image, ImageBlock, ImageDocument, or a file path as a string. Defaults to None.
            query_image_path (Optional[os.PathLike[str]]): The file path to an image to use as part of the query. Defaults to None.

        Returns:
            RESPONSE_TYPE: The result of the query.

        Notes:
            - At least one of `query_str`, `query_image`, or `query_image_path` should be provided.

        """
        qb = ExtendedQueryBundle(
            query_str=query_str, image_path=query_image_path, image=query_image
        )
        dispatcher.event(QueryStartEvent(query=qb))
        with self.callback_manager.as_trace("query"):
            if not query_str:
                query_str = ""
            query_result = self._query(qb)
        dispatcher.event(QueryEndEvent(query=qb, response=query_result))
        return query_result

    @override
    @dispatcher.span
    async def aquery(
        self,
        query_str: Optional[str] = None,
        query_image: Optional[
            Union[Image.Image, ImageBlock, ImageDocument, str]
        ] = None,
        query_image_path: Optional[os.PathLike[str]] = None,
    ) -> RESPONSE_TYPE:
        """
        Asynchronously executes a query against the managed LanceDB index.

        Args:
            query_str (Optional[str]): The text query string to search for. Defaults to None.
            query_image (Optional[Union[Image.Image, ImageBlock, ImageDocument, str]]): An image or image-like object to use as part of the query. Can be a PIL Image, ImageBlock, ImageDocument, or a file path as a string. Defaults to None.
            query_image_path (Optional[os.PathLike[str]]): The file path to an image to use as part of the query. Defaults to None.

        Returns:
            RESPONSE_TYPE: The result of the query.

        Notes:
            - At least one of `query_str`, `query_image`, or `query_image_path` should be provided.

        """
        qb = ExtendedQueryBundle(
            query_str=query_str, image_path=query_image_path, image=query_image
        )
        dispatcher.event(QueryStartEvent(query=qb))
        with self.callback_manager.as_trace("query"):
            if not query_str:
                query_str = ""

            query_result = await self._aquery(qb)
        dispatcher.event(QueryEndEvent(query=qb, response=query_result))
        return query_result
