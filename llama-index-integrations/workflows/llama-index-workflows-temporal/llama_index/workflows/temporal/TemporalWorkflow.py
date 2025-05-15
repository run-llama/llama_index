import asyncio
from dataclasses import dataclass
from datetime import timedelta
import functools
import time
import uuid
from llama_index.core.workflow.context import AbstractContext, T
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.types import RunResultT
from llama_index.core.workflow.workflow import Workflow, Event
from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Worker
from temporalio import workflow, activity
from typing import Any, Callable, Concatenate, Coroutine, Dict, List, ParamSpec, Type, TypeVar, Union, Optional, get_type_hints, get_origin, get_args
import inspect

@dataclass
class WorkflowContextGet:
    key: str

@dataclass
class WorkflowContextSet:
    key: str
    value: Any

@dataclass
class WorkflowContextSendEvent:
    event: Event

@dataclass
class WorkflowContextSendEvent:
    event: Event
    step: Optional[str] = None

@dataclass
class WorkflowContextQueryProcessedEvents:
    event_type: Type[Event]

class TemporalContext(AbstractContext):

    def __init__(self, wf_handle: WorkflowHandle, workflow_class: "LlamaIndexTemporalWorkflow"):
        self.wf_handle = wf_handle
        self.workflow_class = workflow_class

    async def get(self, key: str, default: Optional[Any] = None) -> Any:
        return (await self.wf_handle.query(self.workflow_class.get_data, WorkflowContextGet(key=key))) or default

    async def set(self, key: str, value: Any) -> None:
        await self.wf_handle.execute_update(self.workflow_class.set_data, WorkflowContextSet(key=key, value=value))

    async def send_event(self, event: T) -> None:
        await self.wf_handle.signal(self.workflow_class.send_event, WorkflowContextSendEvent(event=event))

    async def wait_for_event(
        self,
        event_type: Type[T],
        waiter_event: Optional[Event] = None,
        waiter_id: Optional[str] = None,
        requirements: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 2000
    ) -> T:
        start_time = time.time()
        if waiter_event:
            event_str = _get_full_path(event_type)
            requirements_str = str(requirements)
            waiter_id = waiter_id or f"waiter_{event_str}_{requirements_str}"
            # TODO non-atomic race conditions here, and in general, this is not great pattern for this kind of thing
            created = await self.get(waiter_id, False)
            if not created:
                self.send_event(waiter_event)
                await self.set(waiter_id, True)
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for event {event_type} after {timeout} seconds")
            events = await self.wf_handle.query(self.workflow_class.get_processed_events, WorkflowContextQueryProcessedEvents(event_type=event_type))
            for event in events:
                if event:
                    if all(
                        event.get(k, None) == v
                        for k, v in (requirements or {}).items()
                    ):
                        return event


def build_wf_activities(wf: Workflow) -> List["Activity"]:
    return _build_activities(NamedCallable.from_dict(wf._get_steps()))


def _build_activities(functions: List["NamedCallable"]) -> List["Activity"]:
    """
    Each callable accepts an Event sub type, or a Union of Event types.
    This introspects the callable signatures to identify what Event types they handle
    and returns a mapping of Event types to their respective handlers.
    """
    activities: List["Activity"] = []

    for func in functions:
        callable_func = func.callable
        sig = inspect.signature(callable_func)
        type_hints = get_type_hints(callable_func)

        # Look for parameters that are Event types or Unions of Event types
        for param_name, param in sig.parameters.items():
            # Skip self or cls parameters
            if param_name in ("self", "cls"):
                continue

            # Get the annotated type for this parameter
            param_type = type_hints.get(param_name, param.annotation)

            # Skip parameters without type annotations
            if param_type is inspect.Parameter.empty:
                continue

            # Handle Union types (including Optional which is Union[T, None])
            if get_origin(param_type) == Union:
                event_types = [
                    t for t in get_args(param_type)
                    if t is not type(None) and (t == Event or (inspect.isclass(t) and issubclass(t, Event)))
                ]

                # Add a mapping for each Event type in the Union
                for event_type in event_types:
                    activities.append(Activity.from_callable(func, event_type))

            # Handle direct Event types
            elif param_type == Event or (inspect.isclass(param_type) and issubclass(param_type, Event)):
                activities.append(Activity.from_callable(func, param_type))

    return activities

@dataclass(frozen=True)
class Activity:
    name: str
    event_type: Type[Event]
    callable: Callable

    @property
    def unique_name(self) -> str:
        return Activity.create_unique_name(self.name, self.event_type)

    @classmethod
    def create_unique_name(cls, name: str, event_type: Type[Event]) -> str:
        return f"{name}_{_get_full_path(event_type)}"

    @classmethod
    def from_callable(cls, callable: "NamedCallable", event_type: Type[Event]) -> "Activity":
        name = cls.create_unique_name(callable.name, event_type)

        wrapper = activity.defn(name=name)(_with_dependencies()(callable.callable))
        return cls(name=callable.name, event_type=event_type, callable=wrapper)

P = ParamSpec("P")
R = TypeVar("R")

def _with_dependencies():
    def only_event(fn: Callable[Concatenate[T, P], R]) -> Callable[[T], R]:
        @functools.wraps(fn)
        async def wrapper(event: T):
            # TODO - hydrate the context within the worker
            print("runnin'", event)
            return await fn(event)
        orig_sig      = inspect.signature(fn)

        wrapper.__signature__ = inspect.signature(fn).replace(parameters=[
            next(iter(orig_sig.parameters.values()))
        ])
        return wrapper
    return only_event

@dataclass(frozen=True)
class NamedCallable:
    name: str
    callable: Callable

    @classmethod
    def from_dict(cls, d: Dict[str, Callable]) -> List["NamedCallable"]:
        items = [NamedCallable(name=name, callable=callable) for name, callable in d.items()]
        items.sort(key=lambda x: x.name)
        return items

class LlamaIndexTemporalWorkflow():


    pending_events: List[Event] = []
    _data = {}
    _processed_events = []
    _is_complete = False
    _result: Optional[RunResultT] = None
    _activities: List["Activity"] = []

    def __init__(self, activities: List["Activity"]):
        self._activities = activities

    @workflow.run
    async def run(self, start_event: StartEvent):
        print("starting workflow")
        self.pending_events = [start_event]
        while not self._is_complete:
            print(f"pending events: {len(self.pending_events)}")
            await workflow.wait_condition(lambda: len(self.pending_events) > 0)
            coroutines: List[Coroutine[Any, Any, List[Event]]] = []
            while len(self.pending_events):
                event = self.pending_events.pop(0)
                print(f"processing event {type(event)}")
                self._processed_events.append(event)
                coroutines.extend(self._trigger_from_event(event))
            # this gather and nice ordered processing
            # is being deterministic for temporal's sake here, keeping everything nice and tidy and
            # in order, however this is not doing things like early quitting when a workflow
            # done is received, or triggering subsequent activities early once a new event is received.
            # Can we do that somehow and keep temporal determinism happy?
            new_events_grouped = await asyncio.gather(*coroutines)
            new_events = [event for events in new_events_grouped for event in events]
            self.pending_events.extend(new_events)
            if any(isinstance(event, StopEvent) for event in new_events):
                self._is_complete = True

    def _trigger_from_event(self, event: Event) -> List[Coroutine[Any, Any, List[Event]]]:
        print(f"triggering from event {event} against {len(self._activities)} activities")
        coroutines = []
        for activity in self._activities:
            print(f"type[{type(event)}] isinstance {activity.event_type} = {isinstance(event, activity.event_type)}")
            if isinstance(event, activity.event_type):
                async def exec(activity: "Activity") -> List[Event]:
                    # TODO - somehow hydrate the context within the worker
                    print(f"executing activity {activity.unique_name} with event {event}")
                    maybe_event = await workflow.execute_activity(
                        activity.callable, event, start_to_close_timeout=timedelta(seconds=10)
                    )
                    print(f"maybe event: {maybe_event}")
                    events = []
                    if isinstance(maybe_event, Event):
                        events.append(maybe_event)
                    return events

                coroutines.append(exec(activity))
        return coroutines

    @workflow.query
    async def get_processed_events(self, msg: WorkflowContextQueryProcessedEvents) -> List[Event]:
        return self._processed_events

    @workflow.signal
    def send_event(self, msg: WorkflowContextSendEvent) -> None:
        self._pending_events.append(msg.event)

    @workflow.query
    async def get_result(self) -> RunResultT:
        return self._result

    @workflow.query
    async def get_data(self, msg: WorkflowContextGet) -> Any:
        return self._data[msg.key]

    @workflow.update
    async def set_data(self, msg: WorkflowContextSet) -> None:
        self._data[msg.key] = msg.value



def _get_full_path(ev_type: Type[Event]) -> str:
    qualified_module = ev_type.__module__.replace("__temporal_main__", "__main__")
    return f"{qualified_module}.{ev_type.__name__}"


class TemporalWorkflowHandle(asyncio.Future[RunResultT]):

    pass


class TemporalWorkflowBuilder():
    def __init__(self, client: Client, temporal_workflow: Type[LlamaIndexTemporalWorkflow], instance: Workflow, register_worker: bool = True):
        self._client = client
        self._wf = instance
        self._temporal_workflow = temporal_workflow
        steps = NamedCallable.from_dict(self._wf._get_steps())
        self.activites = _build_activities(steps)
        self._worker = asyncio.create_task(self._create_worker()) if register_worker else None

    async def _create_worker(self):
        pass
        # async with Worker(
        #     self._client,
        #     task_queue=self._wf.__class__.__qualname__,
        #     workflows=[self._temporal_workflow],
        #     activities=[x.callable for x in self.activites]
        # ):
        #     print("created worker")
        #     await asyncio.Future()


    async def run(
        self,
        start_event: Optional[StartEvent] = None,
        **kwargs: Any,
    ) -> TemporalWorkflowHandle:
        print(f"running workflow {self._wf.__class__.__qualname__}")
        print(f"registering activities {[x.unique_name for x in self.activites]}")
        async with Worker(
            self._client,
            task_queue=self._wf.__class__.__qualname__,
            workflows=[self._temporal_workflow],
            activities=[x.callable for x in self.activites]
        ):
            result = await self._client.execute_workflow(
                self._temporal_workflow.run, start_event or self._wf._get_start_event_instance(None, **kwargs),
                id=uuid.uuid4().hex,
                task_queue=self._wf.__class__.__qualname__,
            )
            return result
