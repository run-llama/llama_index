from typing import Any, Dict, List, Optional, Sequence
import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.retrievers.router_retriever import RouterRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools.retriever_tool import RetrieverTool
from llama_index.core.tools.types import ToolMetadata


class _FixedRetriever(BaseRetriever):
    def __init__(self, nodes: List[NodeWithScore]) -> None:
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return list(self._nodes)


class _FixedSelector(BaseSelector):
    def __init__(self, indices: List[int]) -> None:
        self._indices = indices
        super().__init__()

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        return MultiSelection(
            selections=[
                SingleSelection(index=i, reason=f"reason-{i}") for i in self._indices
            ]
        )

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        return self._select(choices, query)


class _EventCaptureHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.end_payloads: List[Dict[str, Any]] = []

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.RETRIEVE and payload is not None:
            self.end_payloads.append(payload)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass


def test_router_retriever_callback_manager_and_event_nodes_list() -> None:
    handler = _EventCaptureHandler()
    cb_manager = CallbackManager([handler])
    n1 = NodeWithScore(node=TextNode(id_="n1", text="alpha"), score=0.9)
    n2 = NodeWithScore(node=TextNode(id_="n2", text="beta"), score=0.8)

    tools = [
        RetrieverTool.from_defaults(retriever=_FixedRetriever([n1]), description="r1"),
        RetrieverTool.from_defaults(retriever=_FixedRetriever([n2]), description="r2"),
    ]
    router = RouterRetriever.from_defaults(
        retriever_tools=tools,
        llm=MockLLM(),
        selector=_FixedSelector([0, 1]),
        callback_manager=cb_manager,
        verbose=True,
    )

    assert router.callback_manager is cb_manager
    assert router._verbose is True

    results = router.retrieve("query")
    assert [r.node.node_id for r in results] == ["n1", "n2"]
    assert len(handler.end_payloads) >= 2
    for payload in handler.end_payloads:
        assert isinstance(payload[EventPayload.NODES], list)


@pytest.mark.asyncio
async def test_router_retriever_aretrieve_event_nodes_list() -> None:
    handler = _EventCaptureHandler()
    cb_manager = CallbackManager([handler])
    n1 = NodeWithScore(node=TextNode(id_="n1", text="alpha"), score=0.9)

    tools = [
        RetrieverTool.from_defaults(retriever=_FixedRetriever([n1]), description="r1"),
    ]
    router = RouterRetriever(
        selector=_FixedSelector([0]),
        retriever_tools=tools,
        llm=MockLLM(),
        callback_manager=cb_manager,
    )

    results = await router.aretrieve("query")
    assert [r.node.node_id for r in results] == ["n1"]
    assert len(handler.end_payloads) >= 2
    for payload in handler.end_payloads:
        assert isinstance(payload[EventPayload.NODES], list)
