import asyncio
import functools
import warnings
from typing import Any, Callable, Dict, Optional, AsyncGenerator, Set

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.workflow.decorators import StepConfig, step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.utils import (
    get_steps_from_class,
    get_steps_from_instance,
    ServiceDefinition,
)

from .context import Context
from .errors import (
    WorkflowDone,
    WorkflowTimeoutError,
    WorkflowValidationError,
    WorkflowRuntimeError,
)
from .service import ServiceManager
from .session import WorkflowSession

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
        service_manager: Optional[ServiceManager] = None,
    ) -> None:
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        # Broker machinery
        self._sessions: Set[WorkflowSession] = set()
        self._step_session: Optional[WorkflowSession] = None
        # Services management
        self._service_manager = service_manager or ServiceManager()

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        # In the typical streaming use case, `run()` is not awaited but wrapped in a asyncio.Task. Since we'll be
        # consuming events produced by `run()`, we must give its Task the chance to run before entering the dequeueing
        # loop.
        await asyncio.sleep(0)

        if len(self._sessions) > 1:
            # We can't possibly know from what session we should stream events, raise an error.
            msg = (
                "This workflow has multiple session running concurrently and cannot stream events. "
                "To be able to stream events, make sure you call `run()` on this workflow only once."
            )
            raise WorkflowRuntimeError(msg)

        # Enter the dequeuing loop.
        session = next(iter(self._sessions))
        while True:
            ev = await session.streaming_queue.get()
            if type(ev) is StopEvent:
                break

            yield ev

    @classmethod
    def add_step(cls, func: Callable) -> None:
        """Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        if func.__name__ in {**get_steps_from_class(cls), **cls._step_functions}:
            msg = f"A step {func.__name__} is already part of this workflow, please choose another name."
            raise WorkflowValidationError(msg)

        cls._step_functions[func.__name__] = func

    def add_workflows(self, **workflows: "Workflow") -> None:
        """Adds one or more nested workflows to this workflow.

        This method only accepts keyword arguments, and the name of the parameter
        will be used as the name of the workflow.
        """
        for name, wf in workflows.items():
            self._service_manager.add(name, wf)

    def _get_steps(self) -> Dict[str, Callable]:
        """Returns all the steps, whether defined as methods or free functions."""
        return {**get_steps_from_instance(self), **self._step_functions}

    def _start(self, stepwise: bool = False) -> WorkflowSession:
        """Sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        session = WorkflowSession(self)
        self._sessions.add(session)

        for name, step_func in self._get_steps().items():
            session._queues[name] = asyncio.Queue()
            session._step_flags[name] = asyncio.Event()
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )
            if not step_config:
                raise ValueError(f"Step {name} is missing `@step` decorator.")

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
                        await session._step_flags[name].wait()

                        # clear all flags so that we only run one step
                        for flag in session._step_flags.values():
                            flag.clear()

                    if self._verbose and name != "_done":
                        print(f"Running step {name}")

                    # run step
                    kwargs = {}
                    if config.context_parameter:
                        kwargs[config.context_parameter] = session.get_context(name)
                    for service_definition in config.requested_services:
                        service = self._service_manager.get(
                            service_definition.name, service_definition.default_value
                        )
                        kwargs[service_definition.name] = service
                    kwargs[config.event_name] = ev

                    # - check if its async or not
                    # - if not async, run it in an executor
                    instrumented_step = dispatcher.span(step)

                    if asyncio.iscoroutinefunction(step):
                        new_ev = await instrumented_step(**kwargs)
                    else:
                        run_task = functools.partial(instrumented_step, **kwargs)
                        new_ev = await asyncio.get_event_loop().run_in_executor(
                            None, run_task
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
                    session._accepted_events.append((name, type(ev).__name__))

                    if not isinstance(new_ev, Event):
                        warnings.warn(
                            f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                        )
                    else:
                        session.send_event(new_ev)

            for _ in range(step_config.num_workers):
                session._tasks.add(
                    asyncio.create_task(
                        _task(name, session._queues[name], step_func, step_config),
                        name=name,
                    )
                )
        return session

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        msg = (
            "Use a Context instance to send events from a step. "
            "Make sure your step method or function takes a parameter of type Context like `ctx: Context` and "
            "replace `self.send_event(...)` with `ctx.session.send_event(...)` in your code."
        )

        if len(self._sessions) > 1:
            # We can't possibly know to what session we should send this event, raise an error.
            raise WorkflowRuntimeError(msg)

        # Emit a warning as this won't work for multiple run()s.
        warnings.warn(msg)
        session = next(iter(self._sessions))
        session.send_event(message=message, step=step)

    @dispatcher.span
    async def run(self, **kwargs: Any) -> str:
        """Runs the workflow until completion.

        Works by
        1. validating the workflow
        2. starting the workflow by setting up the queues and tasks
        3. sending a StartEvent to kick things off
        4. waiting for all tasks to finish or be cancelled
        """
        # Validate the workflow if needed
        self._validate()

        # Start the machinery in a new session
        session = self._start()

        # Send the first event
        session.send_event(StartEvent(**kwargs))

        done, unfinished = await asyncio.wait(
            session._tasks, timeout=self._timeout, return_when=asyncio.FIRST_EXCEPTION
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

        # Bubble up the error if any step raised an exception
        if exception_raised:
            raise exception_raised

        # Raise WorkflowTimeoutError if the workflow timed out
        if not we_done:
            msg = f"Operation timed out after {self._timeout} seconds"
            raise WorkflowTimeoutError(msg)

        return session._retval

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
        # Check if we need to start a new session
        if self._step_session is None:
            self._validate()
            self._step_session = self._start(stepwise=True)
            # Run the first step
            self._step_session.send_event(StartEvent(**kwargs))

        # Unblock all pending steps
        for flag in self._step_session._step_flags.values():
            flag.set()

        # Yield back control to the event loop to give an unblocked step
        # the chance to run (we won't actually sleep here).
        await asyncio.sleep(0)

        # See if we're done, or if a step raised any error
        we_done = False
        exception_raised = None
        for t in self._step_session._tasks:
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

        retval = None
        if we_done:
            # Remove any reference to the tasks
            for t in self._step_session._tasks:
                t.cancel()
                await asyncio.sleep(0)
            retval = self._step_session._retval
            self._step_session = None

        if exception_raised:
            raise exception_raised

        return retval

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return self._step_session is None

    @step
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        ctx.session._retval = ev.result or None
        ctx.session.write_event_to_stream(ev)

        # Signal we want to stop the workflow
        raise WorkflowDone

    def _validate(self) -> None:
        """Validate the workflow to ensure it's well-formed."""
        if self._disable_validation:
            return

        produced_events: Set[type] = {StartEvent}
        consumed_events: Set[type] = set()
        requested_services: Set[ServiceDefinition] = set()

        for name, step_func in self._get_steps().items():
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )
            if not step_config:
                raise ValueError(f"Step {name} is missing `@step` decorator.")

            for event_type in step_config.accepted_events:
                consumed_events.add(event_type)

            for event_type in step_config.return_types:
                if event_type == type(None):
                    # some events may not trigger other events
                    continue

                produced_events.add(event_type)

            requested_services.update(step_config.requested_services)

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

        # Check all the requested services are available
        required_service_names = {
            sd.name for sd in requested_services if sd.default_value is None
        }
        if required_service_names:
            avail_service_names = set(self._service_manager._services.keys())
            missing = required_service_names - avail_service_names
            if missing:
                msg = f"The following services are not available: {', '.join(str(m) for m in missing)}"
                raise WorkflowValidationError(msg)
