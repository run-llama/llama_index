import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.core.workflow.decorators import step, StepConfig
from llama_index.core.workflow.events import StartEvent, StopEvent, Event
from llama_index.core.workflow.utils import get_steps_from_class


from .errors import WorkflowRuntimeError, WorkflowTimeoutError, WorkflowValidationError


class Workflow:
    def __init__(
        self,
        timeout: int = 10,
        disable_validation: bool = False,
        verbose: bool = False,
    ) -> None:
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        # Broker machinery
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._broker_log: List[Event] = []
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._accepted_events: List[Tuple[str, str]] = []
        self._retval: Any = None

    def _start(self, stepwise: bool = False) -> None:
        """Sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        for name, step_func in get_steps_from_class(self).items():
            self._queues[name] = asyncio.Queue()
            self._step_flags[name] = asyncio.Event()
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )
            if not step_config:
                raise ValueError(f"Step {name} is missing `@step()` decorator.")

            async def _task(
                name: str,
                queue: asyncio.Queue,
                step: Callable,
                config: StepConfig,
            ) -> None:
                while True:
                    ev = await queue.get()
                    if type(ev) not in config.accepted_events:
                        continue

                    # do we need to wait for the step flag?
                    if stepwise:
                        await self._step_flags[name].wait()

                        # clear all flags so that we only run one step
                        for flag in self._step_flags.values():
                            flag.clear()

                    if self._verbose:
                        print(f"Running step {name}")

                    # run step
                    new_ev = await step(ev)

                    if self._verbose:
                        print(f"Step {name} produced event {type(new_ev).__name__}")

                    # handle the return value
                    if new_ev is None:
                        continue

                    # Store the accepted event for the drawing operations
                    self._accepted_events.append((name, type(ev).__name__))

                    if not isinstance(new_ev, Event):
                        warnings.warn(
                            f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                        )
                    else:
                        self.send_event(new_ev)

            self._tasks.add(
                asyncio.create_task(
                    _task(name, self._queues[name], step_func, step_config)
                )
            )

    def send_event(self, message: Event) -> None:
        """Sends an event to a specific step in the workflow.

        Currently we send all the events to all the receivers and we let
        them discard events they don't want. This should be optimized so
        that we efficiently send events where we know won't be discarded.
        """
        for queue in self._queues.values():
            queue.put_nowait(message)
        self._broker_log.append(message)

    async def run(self, **kwargs: Any) -> str:
        """Runs the workflow until completion.

        Works by
        1. validating the workflow
        2. starting the workflow by setting up the queues and tasks
        3. sending a StartEvent to kick things off
        4. waiting for all tasks to finish or be cancelled
        """
        if self._tasks:
            msg = "Workflow is already running, wait for it to finish before running again."
            raise WorkflowRuntimeError(msg)

        # Reset the events log
        self._accepted_events = []
        # Validate the workflow if needed
        self._validate()
        # Start the machinery
        self._start()
        # Send the first event
        self.send_event(StartEvent(kwargs))

        _, unfinished = await asyncio.wait(self._tasks, timeout=self._timeout)
        if unfinished:
            msg = f"Operation timed out after {self._timeout} seconds"
            raise WorkflowTimeoutError(msg)

        return self._retval

    async def run_step(self, **kwargs: Any) -> Optional[str]:
        """Runs the workflow stepwise until completion.

        Works by
        1. Validating and setting up the queues and tasks if the first step hasn't been started
        2. Sending a StartEvent to kick things off
        3. Sets the flag for all steps to run once (if they can run)
        4. Waiting for the next step(s) to finish
        5. Returning the result if the workflow is done
        """
        # Check if we need to start
        if not self._tasks:
            self._accepted_events = []
            self._validate()
            self._start(stepwise=True)
            # Run the first step
            self.send_event(StartEvent(kwargs))

        # Unblock all pending steps
        for flag in self._step_flags.values():
            flag.set()

        # Yield back control to the event loop to give an unblocked step
        # the chance to run (we won't actually sleep here).
        await asyncio.sleep(0)

        # If we're done, return the result
        if self.is_done:
            return self._retval

        return None

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return len(self._tasks) == 0

    def get_result(self) -> Any:
        """Returns the result of the workflow."""
        return self._retval

    @step()
    async def _done(self, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        # Stop all the tasks
        for t in self._tasks:
            t.cancel()
        # Remove any reference to the tasks
        self._tasks = set()
        self._retval = ev.result or None

    def _validate(self) -> None:
        """Validate the workflow to ensure it's well-formed."""
        if self._disable_validation:
            return

        produced_events: Set[type] = {StartEvent}
        consumed_events: Set[type] = set()

        for name, step_func in get_steps_from_class(self).items():
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )
            if not step_config:
                raise ValueError(f"Step {name} is missing `@step()` decorator.")

            for event_type in step_config.accepted_events:
                consumed_events.add(event_type)

            for event_type in step_config.return_types:
                if event_type == type(None):
                    # some events may not trigger other events
                    continue

                produced_events.add(event_type)

        # Check if all consumed events are produced
        unconsumed_events = consumed_events - produced_events
        if unconsumed_events:
            raise WorkflowValidationError(
                f"The following events are consumed but never produced: {unconsumed_events}"
            )

        # Check if there are any unused produced events (except StopEvent)
        unused_events = produced_events - consumed_events - {StopEvent}
        if unused_events:
            raise WorkflowValidationError(
                f"The following events are produced but never consumed: {unused_events}"
            )

        # Check if there's at least one step that consumes StartEvent
        if StartEvent not in consumed_events:
            raise WorkflowValidationError("No step consumes StartEvent")

        # Check if there's at least one step that produces StopEvent
        if StopEvent not in produced_events:
            raise WorkflowValidationError("No step produces StopEvent")
