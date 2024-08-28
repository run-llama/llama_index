import asyncio
from typing import Any, TYPE_CHECKING, Dict, Set, List, Tuple, Optional


from .context import Context
from .decorators import StepConfig
from .events import Event
from .errors import WorkflowRuntimeError

if TYPE_CHECKING:
    from .workflow import Workflow


class WorkflowSession:
    def __init__(self, workflow: "Workflow") -> None:
        self._workflow = workflow
        # Broker machinery
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._broker_log: List[Event] = []
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._accepted_events: List[Tuple[str, str]] = []
        self._retval: Any = None
        self._root_context = Context(self)
        # Context management
        self._step_to_context: Dict[str, Context] = {}
        # Streaming machinery
        self._streaming_queue: asyncio.Queue = asyncio.Queue()

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

    def get_context(self, step_name: str) -> Context:
        """Get the global context for this workflow.

        The Workflow instance is ultimately responsible for managing the lifecycle
        of the global context object and for passing it to the steps functions that
        require it.
        """
        if step_name not in self._step_to_context:
            self._step_to_context[step_name] = Context(parent=self._root_context)
        return self._step_to_context[step_name]

    def get_result(self) -> Any:
        """Returns the result of the workflow."""
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        return self._streaming_queue
