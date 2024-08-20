import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.workflow.decorators import StepConfig, step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)

from .context import Context
from .errors import (
    WorkflowDone,
    WorkflowRuntimeError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)

dispatcher = get_dispatcher(__name__)


class _WorkflowMeta(type):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step_functions: Dict[str, Callable] = {}


class Workflow(metaclass=_WorkflowMeta):
    def __init__(
        self,
        timeout: Optional[float] = 10.0,
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
        # Context management
        self._root_context: Context = Context()
        self._step_to_context: Dict[str, Context] = {}

    @classmethod
    def add_step(cls, func: Callable) -> None:
        """Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        if func.__name__ in {**get_steps_from_class(cls), **cls._step_functions}:
            msg = f"A step {func.__name__} is already part of this workflow, please choose another name."
            raise WorkflowValidationError(msg)

        cls._step_functions[func.__name__] = func

    def get_context(self, step_name: str) -> Context:
        """Get the global context for this workflow.

        The Workflow instance is ultimately responsible for managing the lifecycle
        of the global context object and for passing it to the steps functions that
        require it.
        """
        if step_name not in self._step_to_context:
            self._step_to_context[step_name] = Context(parent=self._root_context)
        return self._step_to_context[step_name]

    def _get_steps(self) -> Dict[str, Callable]:
        """Returns all the steps, whether defined as methods or free functions."""
        return {**get_steps_from_instance(self), **self._step_functions}

    def _start(self, stepwise: bool = False) -> None:
        """Sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        for name, step_func in self._get_steps().items():
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

                    if self._verbose and name != "_done":
                        print(f"Running step {name}")

                    # run step
                    args = []
                    if config.pass_context:
                        args.append(self.get_context(name))
                    args.append(ev)

                    # - check if its async or not
                    # - if not async, run it in an executor
                    instrumented_step = dispatcher.span(step)

                    if asyncio.iscoroutinefunction(step):
                        new_ev = await instrumented_step(*args)
                    else:
                        new_ev = await asyncio.get_event_loop().run_in_executor(
                            None, instrumented_step, *args
                        )

                    if self._verbose and name != "_done":
                        if new_ev is not None:
                            print(f"Step {name} produced event {type(new_ev).__name__}")
                        else:
                            print(f"Step {name} produced no event")

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

            for _ in range(step_config.num_workers):
                self._tasks.add(
                    asyncio.create_task(
                        _task(name, self._queues[name], step_func, step_config),
                        name=name,
                    )
                )

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        """Sends an event to a specific step in the workflow.

        If step is None, the event is sent to all the receivers and we let
        them discard events they don't want.
        """
        if step is None:
            for queue in self._queues.values():
                queue.put_nowait(message)
        else:
            if step not in self._get_steps():
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_func = self._get_steps()[step]
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

    @dispatcher.span
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
        self.send_event(StartEvent(**kwargs))

        done, unfinished = await asyncio.wait(
            self._tasks, timeout=self._timeout, return_when=asyncio.FIRST_EXCEPTION
        )

        we_done = False
        exception_raised = None
        # A task that raised an exception will be returned in the `done` set
        for task in done:
            # Check if any exception was raised from a step function
            e = task.exception()
            # If the error was of type WorkflowDone, the _done step run successfully
            if type(e) == WorkflowDone:
                we_done = True
            # In any other case, we will re-raise after cleaning up.
            # Since wait() is called with return_when=asyncio.FIRST_EXCEPTION,
            # we can assume exception_raised will be only one.
            elif e is not None:
                exception_raised = e
                break

        # Cancel any pending tasks
        for t in unfinished:
            t.cancel()
            await asyncio.sleep(0)

        # Remove any reference to the tasks
        self._tasks = set()

        # Bubble up the error if any step raised an exception
        if exception_raised:
            raise exception_raised

        # Raise WorkflowTimeoutError if the workflow timed out
        if not we_done:
            msg = f"Operation timed out after {self._timeout} seconds"
            raise WorkflowTimeoutError(msg)

        return self._retval

    @dispatcher.span
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
            self.send_event(StartEvent(**kwargs))

        # Unblock all pending steps
        for flag in self._step_flags.values():
            flag.set()

        # Yield back control to the event loop to give an unblocked step
        # the chance to run (we won't actually sleep here).
        await asyncio.sleep(0)

        # See if we're done, or if a step raised any error
        we_done = False
        exception_raised = None
        for t in self._tasks:
            if not t.done():
                continue

            e = t.exception()
            if e is None:
                continue

            # Check if we're done
            if type(e) == WorkflowDone:
                we_done = True
                continue

            # In any other case, bubble up the exception
            exception_raised = e

        if we_done:
            # Remove any reference to the tasks
            for t in self._tasks:
                t.cancel()
                await asyncio.sleep(0)
            self._tasks = set()

        if exception_raised:
            raise exception_raised

        return self._retval

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return len(self._tasks) == 0

    def get_result(self) -> Any:
        """Returns the result of the workflow."""
        return self._retval

    @step()
    async def _done(self, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        self._retval = ev.result or None
        # Signal we want to stop the workflow
        raise WorkflowDone

    def _validate(self) -> None:
        """Validate the workflow to ensure it's well-formed."""
        if self._disable_validation:
            return

        produced_events: Set[type] = {StartEvent}
        consumed_events: Set[type] = set()

        for name, step_func in self._get_steps().items():
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
