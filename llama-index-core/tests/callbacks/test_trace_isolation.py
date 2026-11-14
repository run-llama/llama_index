import asyncio
import copy
import gc
import pickle
import weakref
from contextvars import Context, copy_context
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import pytest

from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import base as manager_module
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import BASE_TRACE_EVENT, CBEventType
from llama_index.core.llms.mock import MockLLM


class RecordingHandler(BaseCallbackHandler):
    def __init__(self, fail_start: bool = False, fail_end: bool = False) -> None:
        super().__init__([], [])
        self.maps: Dict[str, Dict[str, List[str]]] = {}
        self.parents: Dict[str, str] = {}
        self.fail_start = fail_start
        self.fail_end = fail_end

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if self.fail_start:
            self.fail_start = False
            raise RuntimeError("start failed")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if self.fail_end:
            self.fail_end = False
            raise RuntimeError("end failed")
        assert trace_id is not None
        self.maps[trace_id] = {
            parent: list(children) for parent, children in (trace_map or {}).items()
        }

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        self.parents[event_id] = parent_id
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        pass


@pytest.fixture(autouse=True)
def isolate_trace_stacks() -> Generator[None, None, None]:
    stack_token = manager_module.global_stack_trace.set([BASE_TRACE_EVENT])
    ids_token = manager_module.global_stack_trace_ids.set([])
    yield
    manager_module.global_stack_trace.reset(stack_token)
    manager_module.global_stack_trace_ids.reset(ids_token)


async def interleave(
    first: CallbackManager, second: CallbackManager, finish_first: str = "A"
) -> Dict[str, Dict[str, List[str]]]:
    a_started, b_started, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    views = {}

    async def run(label: str, manager: CallbackManager) -> None:
        if label == "B":
            await a_started.wait()
        with manager.as_trace(label):
            with manager.event(CBEventType.QUERY, event_id=f"{label}-query"):
                (a_started if label == "A" else b_started).set()
                await b_started.wait()
                if label != finish_first:
                    await finished.wait()
                views[label] = {
                    key: list(children) for key, children in manager.trace_map.items()
                }
        if label == finish_first:
            finished.set()

    await asyncio.gather(run("A", first), run("B", second))
    return views


def assert_independent_roots(handler: RecordingHandler) -> None:
    assert handler.parents == {
        "A-query": BASE_TRACE_EVENT,
        "B-query": BASE_TRACE_EVENT,
    }
    assert handler.maps == {
        "A": {BASE_TRACE_EVENT: ["A-query"]},
        "B": {BASE_TRACE_EVENT: ["B-query"]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_first", ["A", "B"])
async def test_shared_manager_independent_roots(finish_first: str) -> None:
    handler = RecordingHandler()
    manager = CallbackManager([handler])
    views = await interleave(manager, manager, finish_first)
    assert_independent_roots(handler)
    assert views == handler.maps


@pytest.mark.asyncio
async def test_separate_managers_independent_roots() -> None:
    handler = RecordingHandler()
    await interleave(CallbackManager([handler]), CallbackManager([handler]))
    assert_independent_roots(handler)


def test_shared_manager_sequential_roots() -> None:
    handler = RecordingHandler()
    manager = CallbackManager([handler])
    for label in ("A", "B"):
        with manager.as_trace(label):
            with manager.event(CBEventType.QUERY, event_id=f"{label}-query"):
                pass
    assert_independent_roots(handler)
    assert manager.trace_map == {BASE_TRACE_EVENT: ["B-query"]}


@pytest.mark.asyncio
async def test_child_tasks_and_nested_traces_aggregate() -> None:
    handler = RecordingHandler()
    manager = CallbackManager([handler])

    async def child(label: str) -> None:
        with manager.as_trace(label):
            with manager.event(CBEventType.LLM, event_id=label):
                pass

    with manager.as_trace("parent"):
        with manager.event(CBEventType.QUERY, event_id="parent-query"):
            await asyncio.gather(child("child-A"), child("child-B"))
            assert manager.trace_map["parent-query"] == ["child-A", "child-B"]
    assert handler.maps == {
        "parent": {
            BASE_TRACE_EVENT: ["parent-query"],
            "parent-query": ["child-A", "child-B"],
        }
    }


@pytest.mark.parametrize("copy_kind", ["deepcopy", "pickle", "llm_model_copy"])
def test_manager_copy_after_trace(copy_kind: str) -> None:
    manager = CallbackManager([RecordingHandler()])
    with manager.as_trace("completed"):
        with manager.event(CBEventType.QUERY, event_id="completed-query"):
            pass

    if copy_kind == "deepcopy":
        cloned = copy.deepcopy(manager)
    elif copy_kind == "pickle":
        cloned = pickle.loads(pickle.dumps(manager))
    else:
        cloned = (
            MockLLM(callback_manager=manager).model_copy(deep=True).callback_manager
        )

    assert cloned is not manager
    assert (
        cloned.trace_map == manager.trace_map == {BASE_TRACE_EVENT: ["completed-query"]}
    )
    with cloned.as_trace("cloned"):
        with cloned.event(CBEventType.QUERY, event_id="cloned-query"):
            pass
    assert manager.trace_map == {BASE_TRACE_EVENT: ["completed-query"]}
    assert cloned.trace_map == {BASE_TRACE_EVENT: ["cloned-query"]}


@pytest.mark.parametrize("phase", ["start", "end"])
def test_handler_failure_allows_following_trace(phase: str) -> None:
    handler = RecordingHandler(fail_start=phase == "start", fail_end=phase == "end")
    manager = CallbackManager([handler])
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        with manager.as_trace("failed"):
            with manager.event(CBEventType.QUERY, event_id="failed-query"):
                pass

    with manager.as_trace("following"):
        with manager.event(CBEventType.QUERY, event_id="following-query"):
            pass
    assert handler.maps["following"] == {BASE_TRACE_EVENT: ["following-query"]}


class EventId(str):
    pass


@pytest.mark.parametrize("fail_end", [False, True])
def test_finished_trace_does_not_retain_event_ids(fail_end: bool) -> None:
    def short_trace() -> weakref.ReferenceType[EventId]:
        event_id = EventId("short-query")
        observed = weakref.ref(event_id)
        manager = CallbackManager([RecordingHandler(fail_end=fail_end)])
        try:
            with manager.as_trace("short"):
                with manager.event(CBEventType.QUERY, event_id=event_id):
                    pass
        except RuntimeError:
            if not fail_end:
                raise
        return observed

    observed = short_trace()
    gc.collect()
    assert observed() is None


def test_copied_context_does_not_retain_finished_manager_events() -> None:
    def capture_context() -> tuple[
        Context, weakref.ReferenceType[EventId], weakref.ReferenceType[CallbackManager]
    ]:
        event_id = EventId("captured-query")
        manager = CallbackManager([RecordingHandler()])
        with manager.as_trace("captured"):
            with manager.event(CBEventType.QUERY, event_id=event_id):
                pass
            inherited = copy_context()
        return inherited, weakref.ref(event_id), weakref.ref(manager)

    inherited, event_id_ref, manager_ref = capture_context()
    gc.collect()
    assert inherited is not None
    assert manager_ref() is None
    assert event_id_ref() is None


@dataclass(eq=True, init=False)
class UnhashableManager(CallbackManager):
    pass


def test_unhashable_manager_subclass_can_trace() -> None:
    handler = RecordingHandler()
    manager = UnhashableManager([handler])
    with pytest.raises(TypeError, match="unhashable"):
        hash(manager)
    with manager.as_trace("subclass"):
        with manager.event(CBEventType.QUERY, event_id="subclass-query"):
            pass
    assert handler.maps == {"subclass": {BASE_TRACE_EVENT: ["subclass-query"]}}
