import asyncio
import warnings
from collections import defaultdict
from typing import Dict, Any, Optional, List, Type, TYPE_CHECKING, Set, Tuple

from .decorators import StepConfig
from .events import Event
from .errors import WorkflowRuntimeError

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow


class Context:
    """A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be available across iterations during a workflow
    execution, and across multiple workflow runs.
    Every context instance offers two type of data storage: a global one, that's shared among all the steps within a
    workflow, and private one, that's only accessible from a single step.

    Both `set` and `get` operations on global data are governed by a lock, and considered coroutine-safe.
    """

    def __init__(self, workflow: "Workflow", stepwise: bool = False) -> None:
        self.stepwise = stepwise
        self._workflow = workflow
        # Broker machinery
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._broker_log: List[Event] = []
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._accepted_events: List[Tuple[str, str]] = []
        self._retval: Any = None
        # Streaming machinery
        self._streaming_queue: asyncio.Queue = asyncio.Queue()
        # Global data storage
        self._lock = asyncio.Lock()
        self._globals: Dict[str, Any] = {}
        # Step-specific instance
        self._events_buffer: Dict[Type[Event], List[Event]] = defaultdict(list)

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

    async def get(self, key: str, default: Optional[Any] = None) -> Any:
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
            elif default is not None:
                return default

        msg = f"Key '{key}' not found in Context"
        raise ValueError(msg)

    @property
    def data(self) -> Dict[str, Any]:
        """This property is provided for backward compatibility.

        Use `get` and `set` instead.
        """
        msg = "`data` is deprecated, please use the `get` and `set` method to store data into the Context."
        warnings.warn(msg, DeprecationWarning)
        return self._globals

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._lock

    @property
    def session(self) -> "Context":
        """This property is provided for backward compatibility."""
        msg = "`session` is deprecated, please use the Context instance directly."
        warnings.warn(msg, DeprecationWarning)
        return self

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[type(ev)].append(ev)

        retval: List[Event] = []
        for e_type in expected:
            e_instance_list = self._events_buffer.get(e_type)
            if e_instance_list:
                retval.append(e_instance_list.pop(0))

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for ev in retval:
            self._events_buffer[type(ev)].append(ev)

        return None

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        """Sends an event to a specific step in the workflow.

        If step is None, the event is sent to all the receivers and we let
        them discard events they don't want.
        """
        if step is None:
            for queue in self._queues.values():
                queue.put_nowait(message)
        else:
            if step not in self._workflow._get_steps():
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_func = self._workflow._get_steps()[step]
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )

            if step_config and type(message) in step_config.accepted_events:
                self._queues[step].put_nowait(message)
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._broker_log.append(message)

    def write_event_to_stream(self, ev: Optional[Event]) -> None:
        self._streaming_queue.put_nowait(ev)

    def get_result(self) -> Any:
        """Returns the result of the workflow."""
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        return self._streaming_queue
