import asyncio
import functools
import json
import time
import uuid
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from llama_index.core.instrumentation.dispatcher import Dispatcher

from .checkpointer import CheckpointCallback
from .context_serializers import BaseSerializer, JsonSerializer
from .decorators import StepConfig
from .errors import WorkflowCancelledByUser, WorkflowDone, WorkflowRuntimeError
from .events import Event, InputRequiredEvent
from .service import ServiceManager
from .types import RunResultT

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow

T = TypeVar("T", bound=Event)


class Context:
    """A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be available across iterations during a workflow
    execution, and across multiple workflow runs.
    Every context instance offers two type of data storage: a global one, that's shared among all the steps within a
    workflow, and private one, that's only accessible from a single step.

    Both `set` and `get` operations on global data are governed by a lock, and considered coroutine-safe.
    """

    def __init__(
        self,
        workflow: "Workflow",
        stepwise: bool = False,
    ) -> None:
        self.stepwise = stepwise
        self.is_running = False
        # Store the step configs of this workflow, to be used in send_event
        self._step_configs: dict[str, Optional[StepConfig]] = {}
        for step_name, step_func in workflow._get_steps().items():
            self._step_configs[step_name] = getattr(step_func, "__step_config", None)

        # Init broker machinery
        self._init_broker_data()

        # Global data storage
        self._lock = asyncio.Lock()
        self._globals: Dict[str, Any] = {}

    def _init_broker_data(self) -> None:
        self._waiter_id = str(uuid.uuid4())
        self._queues: Dict[str, asyncio.Queue] = {self._waiter_id: asyncio.Queue()}
        self._tasks: Set[asyncio.Task] = set()
        self._broker_log: List[Event] = []
        self._cancel_flag: asyncio.Event = asyncio.Event()
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._step_events_holding: Optional[List[Event]] = None
        self._step_lock: asyncio.Lock = asyncio.Lock()
        self._step_condition: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._step_event_written: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._accepted_events: List[Tuple[str, str]] = []
        self._retval: RunResultT = None
        # Map the step names that were executed to a list of events they received.
        # This will be serialized, and is needed to resume a Workflow run passing
        # an existing context.
        self._in_progress: Dict[str, List[Event]] = defaultdict(list)
        # Keep track of the steps currently running. This is only valid when a
        # workflow is running and won't be serialized. Note that a single step
        # might have multiple workers, so we keep a counter.
        self._currently_running_steps: DefaultDict[str, int] = defaultdict(int)
        # Streaming machinery
        self._streaming_queue: asyncio.Queue = asyncio.Queue()
        # Step-specific instance
        self._events_buffer: Dict[str, List[Event]] = defaultdict(list)

    def _serialize_queue(self, queue: asyncio.Queue, serializer: BaseSerializer) -> str:
        queue_items = list(queue._queue)  # type: ignore
        queue_objs = [serializer.serialize(obj) for obj in queue_items]
        return json.dumps(queue_objs)  # type: ignore

    def _deserialize_queue(
        self,
        queue_str: str,
        serializer: BaseSerializer,
        prefix_queue_objs: List[Any] = [],
    ) -> asyncio.Queue:
        queue_objs = json.loads(queue_str)
        queue_objs = prefix_queue_objs + queue_objs
        queue = asyncio.Queue()  # type: ignore
        for obj in queue_objs:
            event_obj = serializer.deserialize(obj)
            queue.put_nowait(event_obj)
        return queue

    def _serialize_globals(self, serializer: BaseSerializer) -> Dict[str, Any]:
        serialized_globals = {}
        for key, value in self._globals.items():
            try:
                serialized_globals[key] = serializer.serialize(value)
            except Exception as e:
                raise ValueError(f"Failed to serialize value for key {key}: {e}")
        return serialized_globals

    def _deserialize_globals(
        self, serialized_globals: Dict[str, Any], serializer: BaseSerializer
    ) -> Dict[str, Any]:
        deserialized_globals = {}
        for key, value in serialized_globals.items():
            try:
                deserialized_globals[key] = serializer.deserialize(value)
            except Exception as e:
                raise ValueError(f"Failed to deserialize value for key {key}: {e}")
        return deserialized_globals

    def to_dict(self, serializer: Optional[BaseSerializer] = None) -> Dict[str, Any]:
        serializer = serializer or JsonSerializer()

        return {
            "globals": self._serialize_globals(serializer),
            "streaming_queue": self._serialize_queue(self._streaming_queue, serializer),
            "queues": {
                k: self._serialize_queue(v, serializer) for k, v in self._queues.items()
            },
            "stepwise": self.stepwise,
            "events_buffer": {
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self._events_buffer.items()
            },
            "in_progress": {
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self._in_progress.items()
            },
            "accepted_events": self._accepted_events,
            "broker_log": [serializer.serialize(ev) for ev in self._broker_log],
            "waiter_id": self._waiter_id,
            "is_running": self.is_running,
        }

    @classmethod
    def from_dict(
        cls,
        workflow: "Workflow",
        data: Dict[str, Any],
        serializer: Optional[BaseSerializer] = None,
    ) -> "Context":
        serializer = serializer or JsonSerializer()

        context = cls(workflow, stepwise=data["stepwise"])
        context._globals = context._deserialize_globals(data["globals"], serializer)
        context._streaming_queue = context._deserialize_queue(
            data["streaming_queue"], serializer
        )
        context._events_buffer = {
            k: [serializer.deserialize(ev) for ev in v]
            for k, v in data["events_buffer"].items()
        }
        if len(context._events_buffer) == 0:
            context._events_buffer = defaultdict(list)
        context._accepted_events = data["accepted_events"]
        context._waiter_id = data.get("waiter_id", str(uuid.uuid4()))
        context._broker_log = [serializer.deserialize(ev) for ev in data["broker_log"]]
        context.is_running = data["is_running"]
        # load back up whatever was in the queue as well as the events whose steps
        # were in progress when the serialization of the Context took place
        context._queues = {
            k: context._deserialize_queue(
                v, serializer, prefix_queue_objs=data["in_progress"].get(k, [])
            )
            for k, v in data["queues"].items()
        }
        context._in_progress = defaultdict(list)
        return context

    async def set(self, key: str, value: Any, make_private: bool = False) -> None:
        """Store `value` into the Context under `key`.

        Args:
            key: A unique string to identify the value stored.
            value: The data to be stored.

        Raises:
            ValueError: When make_private is True but a key already exists in the global storage.
        """
        if make_private:
            warnings.warn(
                "`make_private` is deprecated and will be ignored", DeprecationWarning
            )

        async with self.lock:
            self._globals[key] = value

    async def mark_in_progress(self, name: str, ev: Event) -> None:
        """Add input event to in_progress dict.

        Args:
            name (str): The name of the step that is in progress.
            ev (Event): The input event that kicked off this step.
        """
        async with self.lock:
            self._in_progress[name].append(ev)

    async def remove_from_in_progress(self, name: str, ev: Event) -> None:
        """Remove input event from active steps.

        Args:
            name (str): The name of the step that has been completed.
            ev (Event): The associated input event that kicked of this completed step.
        """
        async with self.lock:
            events = [e for e in self._in_progress[name] if e != ev]
            self._in_progress[name] = events

    async def add_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] += 1

    async def remove_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] -= 1
            if self._currently_running_steps[name] == 0:
                del self._currently_running_steps[name]

    async def running_steps(self) -> List[str]:
        async with self.lock:
            return list(self._currently_running_steps)

    async def get(self, key: str, default: Optional[Any] = Ellipsis) -> Any:
        """Get the value corresponding to `key` from the Context.

        Args:
            key: A unique string to identify the value stored.
            default: The value to return when `key` is missing instead of raising an exception.

        Raises:
            ValueError: When there's not value accessible corresponding to `key`.
        """
        async with self.lock:
            if key in self._globals:
                return self._globals[key]
            elif default is not Ellipsis:
                return default

        msg = f"Key '{key}' not found in Context"
        raise ValueError(msg)

    @property
    def data(self) -> Dict[str, Any]:  # pragma: no cover
        """This property is provided for backward compatibility.

        Use `get` and `set` instead.
        """
        msg = "`data` is deprecated, please use the `get` and `set` method to store data into the Context."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._globals

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._lock

    @property
    def session(self) -> "Context":  # pragma: no cover
        """This property is provided for backward compatibility."""
        msg = "`session` is deprecated, please use the Context instance directly."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self

    def _get_full_path(self, ev_type: Type[Event]) -> str:
        return f"{ev_type.__module__}.{ev_type.__name__}"

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[self._get_full_path(type(ev))].append(ev)

        retval: List[Event] = []
        for e_type in expected:
            e_instance_list = self._events_buffer.get(self._get_full_path(e_type))
            if e_instance_list:
                retval.append(e_instance_list.pop(0))

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for ev in retval:
            self._events_buffer[self._get_full_path(type(ev))].append(ev)

        return None

    def add_holding_event(self, event: Event) -> None:
        """Add an event to the list of those collected in current step.

        This is only relevant for stepwise execution.
        """
        if self.stepwise:
            if self._step_events_holding is None:
                self._step_events_holding = []

            self._step_events_holding.append(event)

    def get_holding_events(self) -> List[Event]:
        """Returns a copy of the list of events holding the stepwise execution."""
        if self._step_events_holding is None:
            return []

        return list(self._step_events_holding)

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        """Sends an event to a specific step in the workflow.

        If step is None, the event is sent to all the receivers and we let
        them discard events they don't want.
        """
        self.add_holding_event(message)

        if step is None:
            for queue in self._queues.values():
                queue.put_nowait(message)
        else:
            if step not in self._step_configs:
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_config = self._step_configs[step]
            if step_config and type(message) in step_config.accepted_events:
                self._queues[step].put_nowait(message)
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._broker_log.append(message)

    async def wait_for_event(
        self,
        event_type: Type[T],
        requirements: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 2000,
    ) -> T:
        """Asynchronously wait for a specific event type to be received.

        Args:
            event_type: The type of event to wait for
            requirements: Optional dict of requirements the event must match
            timeout: Optional timeout in seconds. Defaults to 2000s.

        Returns:
            The event type that was requested.

        Raises:
            asyncio.TimeoutError: If the timeout is reached before receiving matching event
        """
        requirements = requirements or {}

        while True:
            event = await asyncio.wait_for(
                self._queues[self._waiter_id].get(), timeout=timeout
            )
            if type(event) is event_type:
                if all(
                    event.get(k, default=None) == v for k, v in requirements.items()
                ):
                    return event
                else:
                    continue

    def write_event_to_stream(self, ev: Optional[Event]) -> None:
        self._streaming_queue.put_nowait(ev)

    def get_result(self) -> RunResultT:
        """Returns the result of the workflow."""
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        return self._streaming_queue

    def clear(self) -> None:
        """Clear any data stored in the context."""
        # Clear the user data storage
        self._globals.clear()

    async def shutdown(self) -> None:
        """To be called when a workflow ends.

        We clear all the tasks and set the is_running flag. Note that we
        don't clear _globals or _queues so that the context can be still
        used after the shutdown to fetch data or consume leftover events.
        """
        self.is_running = False
        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
        # Wait for all tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def add_step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        stepwise: bool,
        verbose: bool,
        checkpoint_callback: Optional[CheckpointCallback],
        run_id: str,
        service_manager: ServiceManager,
        dispatcher: Dispatcher,
    ) -> None:
        self._tasks.add(
            asyncio.create_task(
                self._step_worker(
                    name=name,
                    step=step,
                    config=config,
                    stepwise=stepwise,
                    verbose=verbose,
                    checkpoint_callback=checkpoint_callback,
                    run_id=run_id,
                    service_manager=service_manager,
                    dispatcher=dispatcher,
                ),
                name=name,
            )
        )

    async def _step_worker(
        self,
        name: str,
        step: Callable,
        config: StepConfig,
        stepwise: bool,
        verbose: bool,
        checkpoint_callback: Optional[CheckpointCallback],
        run_id: str,
        service_manager: ServiceManager,
        dispatcher: Dispatcher,
    ) -> None:
        while True:
            ev = await self._queues[name].get()
            if type(ev) not in config.accepted_events:
                continue

            # do we need to wait for the step flag?
            if stepwise:
                await self._step_flags[name].wait()

                # clear all flags so that we only run one step
                for flag in self._step_flags.values():
                    flag.clear()

            if verbose and name != "_done":
                print(f"Running step {name}")

            # run step
            kwargs: Dict[str, Any] = {}
            if config.context_parameter:
                kwargs[config.context_parameter] = self
            for service_definition in config.requested_services:
                service = service_manager.get(
                    service_definition.name, service_definition.default_value
                )
                kwargs[service_definition.name] = service
            kwargs[config.event_name] = ev

            # wrap the step with instrumentation
            instrumented_step = dispatcher.span(step)

            # - check if its async or not
            # - if not async, run it in an executor
            if asyncio.iscoroutinefunction(step):
                retry_start_at = time.time()
                attempts = 0
                while True:
                    await self.mark_in_progress(name=name, ev=ev)
                    await self.add_running_step(name)
                    try:
                        new_ev = await instrumented_step(**kwargs)
                        kwargs.clear()
                        break  # exit the retrying loop
                    except WorkflowDone:
                        await self.remove_from_in_progress(name=name, ev=ev)
                        raise
                    except Exception as e:
                        if config.retry_policy is None:
                            raise WorkflowRuntimeError(
                                f"Error in step '{name}': {e!s}"
                            ) from e

                        delay = config.retry_policy.next(
                            retry_start_at + time.time(), attempts, e
                        )
                        if delay is None:
                            # We're done retrying
                            raise WorkflowRuntimeError(
                                f"Error in step '{name}': {e!s}"
                            ) from e

                        attempts += 1
                        if verbose:
                            print(
                                f"Step {name} produced an error, retry in {delay} seconds"
                            )
                        await asyncio.sleep(delay)
                    finally:
                        await self.remove_running_step(name)

            else:
                try:
                    run_task = functools.partial(instrumented_step, **kwargs)
                    kwargs.clear()
                    new_ev = await asyncio.get_event_loop().run_in_executor(
                        None, run_task
                    )
                except WorkflowDone:
                    await self.remove_from_in_progress(name=name, ev=ev)
                    raise
                except Exception as e:
                    raise WorkflowRuntimeError(f"Error in step '{name}': {e!s}") from e

            if verbose and name != "_done":
                if new_ev is not None:
                    print(f"Step {name} produced event {type(new_ev).__name__}")
                else:
                    print(f"Step {name} produced no event")

            # Store the accepted event for the drawing operations
            if new_ev is not None:
                self._accepted_events.append((name, type(ev).__name__))

            # Fail if the step returned something that's not an event
            if new_ev is not None and not isinstance(new_ev, Event):
                msg = f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                raise WorkflowRuntimeError(msg)

            if stepwise:
                async with self._step_condition:
                    await self._step_condition.wait()

                    if new_ev is not None:
                        self.add_holding_event(new_ev)
                    self._step_event_written.notify()  # shares same lock

                    await self.remove_from_in_progress(name=name, ev=ev)

                    # for stepwise Checkpoint after handler.run_step() call
                    if checkpoint_callback:
                        await checkpoint_callback(
                            run_id=run_id,
                            ctx=self,
                            last_completed_step=name,
                            input_ev=ev,
                            output_ev=new_ev,
                        )
            else:
                # for regular execution, Checkpoint just before firing the next event
                await self.remove_from_in_progress(name=name, ev=ev)
                if checkpoint_callback:
                    await checkpoint_callback(
                        run_id=run_id,
                        ctx=self,
                        last_completed_step=name,
                        input_ev=ev,
                        output_ev=new_ev,
                    )

                # InputRequiredEvent's are special case and need to be written to the stream
                # this way, the user can access and respond to the event
                if isinstance(new_ev, InputRequiredEvent):
                    self.write_event_to_stream(new_ev)
                elif new_ev is not None:
                    self.send_event(new_ev)

    def add_cancel_worker(self) -> None:
        self._tasks.add(asyncio.create_task(self._cancel_worker()))

    async def _cancel_worker(self) -> None:
        try:
            await self._cancel_flag.wait()
            raise WorkflowCancelledByUser
        except asyncio.CancelledError:
            return
