import asyncio
import functools
import time
import warnings
from typing import Any, Callable, Dict, Optional, AsyncGenerator, Set, Tuple

from llama_index.core.instrumentation import get_dispatcher

from .decorators import StepConfig, step
from .context import Context
from .events import Event, StartEvent, StopEvent
from .errors import *
from .service import ServiceManager
from .utils import (
    get_steps_from_class,
    get_steps_from_instance,
    ServiceDefinition,
)


dispatcher = get_dispatcher(__name__)


class WorkflowMeta(type):
    def __init__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        cls._step_functions: Dict[str, Callable] = {}


class Workflow(metaclass=WorkflowMeta):
    """An event-driven abstraction used to orchestrate the execution of different components called "steps".

    Each step is responsible for handling certain event types and possibly emitting new events. Steps can be "bound"
    when they are defined as methods of the `Workflow` class itself, or "unbound" when they are defined as free
    functions. To define a step, the method or function must be decorated with the `@step` decorator.

    Workflows provide basic validation to catch potential runtime errors as soon as possible. Validation happens once,
    when the workflow starts, and does not produce much overhead. It can be disabled in any case.

    Use an instance of a `Workflow` class to run a workflow and stream events produced during execution. Workflows
    can be run step-by-step, by calling the `run_step` function multiple times until completion.
    """

    def __init__(
        self,
        timeout: Optional[float] = 10.0,
        disable_validation: bool = False,
        verbose: bool = False,
        service_manager: Optional[ServiceManager] = None,
    ) -> None:
        """Create an instance of the workflow.

        Args:
            timeout: number of seconds after the workflow execution will be halted, raising a `WorkflowTimeoutError`
                exception. If set to `None`, the timeout will be disabled.
            disable_validaton: whether or not the workflow should be validated before running. In case the workflow is
                misconfigured, a call to `run` will raise a `WorkflowValidationError` exception explaining the details
                of the problem.
            verbose: whether or not the workflow should print additional informative messages during execution.
            service_manager: The instance of the `ServiceManager` used to make nested workflows available to this
                workflow instance. The default value is the best choice unless you're customizing the workflow runtime.
        """
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        # Broker machinery
        self._contexts: Set[Context] = set()
        self._stepwise_context: Optional[Context] = None
        # Services management
        self._service_manager = service_manager or ServiceManager()

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        """Returns an async generator to consume any event that workflow steps decide to stream.

        To be able to use this generator, the usual pattern is to wrap the `run` call in a background task using
        `asyncio.create_task`, then enter a for loop like this:

            wf = StreamingWorkflow()
            r = asyncio.create_task(wf.run())

            async for ev in wf.stream_events():
                print(ev)

            await r
        """
        # In the typical streaming use case, `run()` is not awaited but wrapped in a asyncio.Task. Since we'll be
        # consuming events produced by `run()`, we must give its Task the chance to run before entering the dequeueing
        # loop.
        await asyncio.sleep(0)

        if len(self._contexts) > 1:
            # We can't possibly know from what session we should stream events, raise an error.
            msg = (
                "This workflow has multiple concurrent runs in progress and cannot stream events. "
                "To be able to stream events, make sure you call `run()` on this workflow only once."
            )
            raise WorkflowRuntimeError(msg)

        # Enter the dequeuing loop.
        ctx = next(iter(self._contexts))
        while True:
            ev = await ctx.streaming_queue.get()
            if type(ev) is StopEvent:
                break

            yield ev

        # remove context to free up room for the next stream_events call
        self._contexts.remove(ctx)

    @classmethod
    def add_step(cls, func: Callable) -> None:
        """Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        step_config: Optional[StepConfig] = getattr(func, "__step_config", None)
        if not step_config:
            msg = f"Step function {func.__name__} is missing the `@step` decorator."
            raise WorkflowValidationError(msg)

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
        return {**get_steps_from_instance(self), **self._step_functions}  # type: ignore[attr-defined]

    def _start(self, stepwise: bool = False) -> Context:
        """Sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        ctx = Context(self)
        self._contexts.add(ctx)

        for name, step_func in self._get_steps().items():
            ctx._queues[name] = asyncio.Queue()
            ctx._step_flags[name] = asyncio.Event()
            # At this point, step_func is guaranteed to have the `__step_config` attribute
            step_config: StepConfig = getattr(step_func, "__step_config")

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
                        await ctx._step_flags[name].wait()

                        # clear all flags so that we only run one step
                        for flag in ctx._step_flags.values():
                            flag.clear()

                    if self._verbose and name != "_done":
                        print(f"Running step {name}")

                    # run step
                    kwargs: Dict[str, Any] = {}
                    if config.context_parameter:
                        kwargs[config.context_parameter] = ctx
                    for service_definition in config.requested_services:
                        service = self._service_manager.get(
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
                            try:
                                new_ev = await instrumented_step(**kwargs)
                                break  # exit the retrying loop
                            except Exception as e:
                                if config.retry_policy is None:
                                    raise e from None

                                delay = config.retry_policy.next(
                                    retry_start_at + time.time(), attempts, e
                                )
                                if delay is None:
                                    # We're done retrying
                                    raise e from None

                                attempts += 1
                                if self._verbose:
                                    print(
                                        f"Step {name} produced an error, retry in {delay} seconds"
                                    )
                                await asyncio.sleep(delay)

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
                    ctx._accepted_events.append((name, type(ev).__name__))

                    if not isinstance(new_ev, Event):
                        warnings.warn(
                            f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                        )
                    else:
                        ctx.send_event(new_ev)

            for _ in range(step_config.num_workers):
                ctx._tasks.add(
                    asyncio.create_task(
                        _task(name, ctx._queues[name], step_func, step_config),
                        name=name,
                    )
                )
        return ctx

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        msg = (
            "Use a Context instance to send events from a step. "
            "Make sure your step method or function takes a parameter of type Context like `ctx: Context` and "
            "replace `self.send_event(...)` with `ctx.send_event(...)` in your code."
        )

        if len(self._contexts) > 1:
            # We can't possibly know to what session we should send this event, raise an error.
            raise WorkflowRuntimeError(msg)

        # Emit a warning as this won't work for multiple run()s.
        warnings.warn(msg)
        ctx = next(iter(self._contexts))
        ctx.send_event(message=message, step=step)

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

        # Start the machinery in a new Context
        ctx = self._start()

        # Send the first event
        ctx.send_event(StartEvent(**kwargs))

        done, unfinished = await asyncio.wait(
            ctx._tasks, timeout=self._timeout, return_when=asyncio.FIRST_EXCEPTION
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
            # Make sure to stop streaming, in case the workflow terminated abnormally
            ctx.write_event_to_stream(StopEvent())
            raise exception_raised

        # Raise WorkflowTimeoutError if the workflow timed out
        if not we_done:
            msg = f"Operation timed out after {self._timeout} seconds"
            raise WorkflowTimeoutError(msg)

        return ctx._retval

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
        if self._stepwise_context is None:
            self._validate()
            self._stepwise_context = self._start(stepwise=True)
            # Run the first step
            self._stepwise_context.send_event(StartEvent(**kwargs))

        # Unblock all pending steps
        for flag in self._stepwise_context._step_flags.values():
            flag.set()

        # Yield back control to the event loop to give an unblocked step
        # the chance to run (we won't actually sleep here).
        await asyncio.sleep(0)

        # See if we're done, or if a step raised any error
        we_done = False
        exception_raised = None
        for t in self._stepwise_context._tasks:
            # Check if we're done
            if not t.done():
                continue

            we_done = True
            e = t.exception()
            if type(e) != WorkflowDone:
                exception_raised = e

        retval = None
        if we_done:
            # Remove any reference to the tasks
            for t in self._stepwise_context._tasks:
                t.cancel()
                await asyncio.sleep(0)
            retval = self._stepwise_context._retval
            self._stepwise_context = None

        if exception_raised:
            raise exception_raised

        return retval

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return self._stepwise_context is None

    @step
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        ctx._retval = ev.result or None
        ctx.write_event_to_stream(ev)

        # Signal we want to stop the workflow
        raise WorkflowDone

    def _validate(self) -> None:
        """Validate the workflow to ensure it's well-formed."""
        if self._disable_validation:
            return

        produced_events: Set[type] = {StartEvent}
        consumed_events: Set[type] = set()
        requested_services: Set[ServiceDefinition] = set()

        for step_func in self._get_steps().values():
            step_config: Optional[StepConfig] = getattr(step_func, "__step_config")
            # At this point we know step config is not None, let's make the checker happy
            assert step_config is not None

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
            names = ", ".join(ev.__name__ for ev in unconsumed_events)
            raise WorkflowValidationError(
                f"The following events are consumed but never produced: {names}"
            )

        # Check if there are any unused produced events (except StopEvent)
        unused_events = produced_events - consumed_events - {StopEvent}
        if unused_events:
            names = ", ".join(ev.__name__ for ev in unused_events)
            raise WorkflowValidationError(
                f"The following events are produced but never consumed: {names}"
            )

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
