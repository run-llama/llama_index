import logging
from typing import Callable, Generator, List, Optional, Sequence, Any

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    PydanticResponse,
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.selectors.utils import get_selector_from_llm
from llama_index.core.settings import Settings
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.utils import print_text

logger = logging.getLogger(__name__)


def combine_responses(
    summarizer: TreeSummarize, responses: List[RESPONSE_TYPE], query_bundle: QueryBundle
) -> RESPONSE_TYPE:
    """Combine multiple response from sub-engines."""
    logger.info("Combining responses from multiple query engines.")

    response_strs = []
    source_nodes = []
    for response in responses:
        if isinstance(response, (StreamingResponse, PydanticResponse)):
            response_obj = response.get_response()
        elif isinstance(response, AsyncStreamingResponse):
            raise ValueError("AsyncStreamingResponse not supported in sync code.")
        else:
            response_obj = response
        source_nodes.extend(response_obj.source_nodes)
        response_strs.append(str(response))

    summary = summarizer.get_response(query_bundle.query_str, response_strs)

    if isinstance(summary, str):
        return Response(response=summary, source_nodes=source_nodes)
    elif isinstance(summary, BaseModel):
        return PydanticResponse(response=summary, source_nodes=source_nodes)
    elif isinstance(summary, Generator):
        return StreamingResponse(response_gen=summary, source_nodes=source_nodes)
    else:
        return AsyncStreamingResponse(response_gen=summary, source_nodes=source_nodes)


async def acombine_responses(
    summarizer: TreeSummarize, responses: List[RESPONSE_TYPE], query_bundle: QueryBundle
) -> RESPONSE_TYPE:
    """Async combine multiple response from sub-engines."""
    logger.info("Combining responses from multiple query engines.")

    response_strs = []
    source_nodes = []
    for response in responses:
        if isinstance(response, (StreamingResponse, PydanticResponse)):
            response_obj = response.get_response()
        elif isinstance(response, AsyncStreamingResponse):
            response_obj = await response.get_response()
        else:
            response_obj = response
        source_nodes.extend(response_obj.source_nodes)
        response_strs.append(str(response))

    summary = await summarizer.aget_response(query_bundle.query_str, response_strs)

    if isinstance(summary, str):
        return Response(response=summary, source_nodes=source_nodes)
    elif isinstance(summary, BaseModel):
        return PydanticResponse(response=summary, source_nodes=source_nodes)
    elif isinstance(summary, Generator):
        return StreamingResponse(response_gen=summary, source_nodes=source_nodes)
    else:
        return AsyncStreamingResponse(response_gen=summary, source_nodes=source_nodes)


class RouterQueryEngine(BaseQueryEngine):
    """
    Router query engine.

    Selects one out of several candidate query engines to execute a query.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
            query engines. They must be wrapped as tools to expose metadata to
            the selector.
        summarizer (Optional[TreeSummarize]): Tree summarizer to summarize sub-results.

    """

    def __init__(
        self,
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: Optional[LLM] = None,
        summarizer: Optional[TreeSummarize] = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm or Settings.llm
        self._selector = selector
        self._query_engines = [x.query_engine for x in query_engine_tools]
        self._metadatas = [x.metadata for x in query_engine_tools]
        self._summarizer = summarizer or TreeSummarize(
            llm=self._llm,
            summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
        )
        self._verbose = verbose

        super().__init__(callback_manager=Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        # NOTE: don't include tools for now
        return {"summarizer": self._summarizer, "selector": self._selector}

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: Optional[LLM] = None,
        selector: Optional[BaseSelector] = None,
        summarizer: Optional[TreeSummarize] = None,
        select_multi: bool = False,
        **kwargs: Any,
    ) -> "RouterQueryEngine":
        llm = llm or Settings.llm

        selector = selector or get_selector_from_llm(llm, is_multi=select_multi)

        assert selector is not None

        return cls(
            selector,
            query_engine_tools,
            llm=llm,
            summarizer=summarizer,
            **kwargs,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            result = self._selector.select(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                responses = []
                for i, engine_ind in enumerate(result.inds):
                    log_str = (
                        f"Selecting query engine {engine_ind}: {result.reasons[i]}."
                    )
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")

                    selected_query_engine = self._query_engines[engine_ind]
                    responses.append(selected_query_engine.query(query_bundle))

                if len(responses) > 1:
                    final_response = combine_responses(
                        self._summarizer, responses, query_bundle
                    )
                else:
                    final_response = responses[0]
            else:
                try:
                    selected_query_engine = self._query_engines[result.ind]
                    log_str = f"Selecting query engine {result.ind}: {result.reason}."
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")
                except ValueError as e:
                    raise ValueError("Failed to select query engine") from e

                final_response = selected_query_engine.query(query_bundle)

            # add selected result
            final_response.metadata = final_response.metadata or {}
            final_response.metadata["selector_result"] = result

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                tasks = []
                for i, engine_ind in enumerate(result.inds):
                    log_str = (
                        f"Selecting query engine {engine_ind}: {result.reasons[i]}."
                    )
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")
                    selected_query_engine = self._query_engines[engine_ind]
                    tasks.append(selected_query_engine.aquery(query_bundle))

                responses = run_async_tasks(tasks)
                if len(responses) > 1:
                    final_response = await acombine_responses(
                        self._summarizer, responses, query_bundle
                    )
                else:
                    final_response = responses[0]
            else:
                try:
                    selected_query_engine = self._query_engines[result.ind]
                    log_str = f"Selecting query engine {result.ind}: {result.reason}."
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")
                except ValueError as e:
                    raise ValueError("Failed to select query engine") from e

                final_response = await selected_query_engine.aquery(query_bundle)

            # add selected result
            final_response.metadata = final_response.metadata or {}
            final_response.metadata["selector_result"] = result

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response


def default_node_to_metadata_fn(node: BaseNode) -> ToolMetadata:
    """
    Default node to metadata function.

    We use the node's text as the Tool description.

    """
    metadata = node.metadata or {}
    if "tool_name" not in metadata:
        raise ValueError("Node must have a tool_name in metadata.")
    return ToolMetadata(name=metadata["tool_name"], description=node.get_content())


class RetrieverRouterQueryEngine(BaseQueryEngine):
    """
    Retriever-based router query engine.

    NOTE: this is deprecated, please use our new ToolRetrieverRouterQueryEngine

    Use a retriever to select a set of Nodes. Each node will be converted
    into a ToolMetadata object, and also used to retrieve a query engine, to form
    a QueryEngineTool.

    NOTE: this is a beta feature. We are figuring out the right interface
    between the retriever and query engine.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
            query engines. They must be wrapped as tools to expose metadata to
            the selector.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_to_query_engine_fn: Callable,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._node_to_query_engine_fn = node_to_query_engine_fn
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        # NOTE: don't include tools for now
        return {"retriever": self._retriever}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes_with_score = self._retriever.retrieve(query_bundle)
        # TODO: for now we only support retrieving one node
        if len(nodes_with_score) > 1:
            raise ValueError("Retrieved more than one node.")

        node = nodes_with_score[0].node
        query_engine = self._node_to_query_engine_fn(node)
        return query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)


class ToolRetrieverRouterQueryEngine(BaseQueryEngine):
    """
    Tool Retriever router query engine.

    Selects a set of candidate query engines to execute a query.

    Args:
        retriever (ObjectRetriever): A retriever that retrieves a set of
            query engine tools.
        summarizer (Optional[TreeSummarize]): Tree summarizer to summarize sub-results.

    """

    def __init__(
        self,
        retriever: ObjectRetriever[QueryEngineTool],
        llm: Optional[LLM] = None,
        summarizer: Optional[TreeSummarize] = None,
    ) -> None:
        llm = llm or Settings.llm
        self._summarizer = summarizer or TreeSummarize(
            llm=llm,
            summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
        )
        self._retriever = retriever

        super().__init__(Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        # NOTE: don't include tools for now
        return {"summarizer": self._summarizer}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            query_engine_tools = self._retriever.retrieve(query_bundle)
            responses = []
            for query_engine_tool in query_engine_tools:
                query_engine = query_engine_tool.query_engine
                responses.append(query_engine.query(query_bundle))

            if len(responses) > 1:
                final_response = combine_responses(
                    self._summarizer, responses, query_bundle
                )
            else:
                final_response = responses[0]

            # add selected result
            final_response.metadata = final_response.metadata or {}
            final_response.metadata["retrieved_tools"] = query_engine_tools

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            query_engine_tools = self._retriever.retrieve(query_bundle)
            tasks = []
            for query_engine_tool in query_engine_tools:
                query_engine = query_engine_tool.query_engine
                tasks.append(query_engine.aquery(query_bundle))
            responses = run_async_tasks(tasks)
            if len(responses) > 1:
                final_response = await acombine_responses(
                    self._summarizer, responses, query_bundle
                )
            else:
                final_response = responses[0]

            # add selected result
            final_response.metadata = final_response.metadata or {}
            final_response.metadata["retrieved_tools"] = query_engine_tools

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response
