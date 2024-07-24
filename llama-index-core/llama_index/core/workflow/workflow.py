import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from llama_index.core.workflow.decorators import step, StepConfig
from llama_index.core.workflow.events import StartEvent, StopEvent, Event
from llama_index.core.workflow.utils import get_steps_from_class


class WorkflowValidationError(Exception):
    pass


class Workflow:
    def __init__(
        self,
        timeout: int = 10,
        disable_validation: bool = False,
        verbose: bool = False,
    ) -> None:
        # configuration
        self._timeout = timeout
        self._disable_validation = disable_validation
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._verbose = verbose

        # state variables
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._events: List[Any] = []
        self._retval = None
        self._event_subscriptions: Dict[type, Set[str]] = defaultdict(set)

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

            for event_types in step_config.accepted_events.values():
                for event_type in event_types:
                    self._event_subscriptions[event_type].add(name)

            async def _task(
                name: str,
                queue: asyncio.Queue,
                step: Callable,
                config: StepConfig,
            ) -> None:
                event_buffer = {name: [] for name in config.accepted_events}
                while True:
                    ev = await queue.get()
                    for param_name, event_types in config.accepted_events.items():
                        for event_type in event_types:
                            if isinstance(ev, event_type):
                                event_buffer[param_name].append(ev)
                                break

                    # do we need to wait for the step flag?
                    if stepwise:
                        await self._step_flags[name].wait()

                        # clear all flags so that we only run one step
                        for flag_name in self._step_flags:
                            self._step_flags[flag_name].clear()

                    kwargs = {**config.kwargs}

                    # pop off and consume the latest event of each type
                    current_events = []
                    for param_name in config.accepted_events:
                        if event_buffer[param_name]:
                            kwargs[param_name] = event_buffer[param_name].pop()
                            current_events.append(type(kwargs[param_name]).__name__)

                    # record the events that were consumed
                    self._events.append((name, current_events))

                    if self._verbose:
                        print(f"Running step {name} with kwargs: {kwargs}")

                    # run step
                    print(f"Running step {name} with kwargs: {kwargs}")
                    new_evs = await step(**kwargs)
                    print(f"Step {name} produced event {new_evs}")

                    # handle the return value
                    if isinstance(new_evs, Event):
                        new_evs = [new_evs]
                    elif new_evs is None:
                        new_evs = []

                    for new_ev in new_evs:
                        if self._verbose:
                            print(f"Step {name} produced event {new_ev}")
                        self.send_event(new_ev)

            self._tasks.add(
                asyncio.create_task(
                    _task(name, self._queues[name], step_func, step_config)
                )
            )

    def send_event(self, message: Event) -> None:
        """Sends an event to a specific step in the workflow."""
        event_type = type(message)
        for step_name in self._event_subscriptions[event_type]:
            self._queues[step_name].put_nowait(message)

    async def run(self, **kwargs: Any) -> str:
        """Runs the workflow until completion.

        Works by
        1. validating the workflow
        2. starting the workflow by setting up the queues and tasks
        3. sending a StartEvent to kick things off
        4. waiting for all tasks to finish or be cancelled
        """
        if not self._disable_validation:
            self._validate()

        self._events = []
        if not self._tasks:
            self._start()

        async with asyncio.timeout(self._timeout):
            self.send_event(StartEvent(kwargs))
            try:
                await asyncio.gather(*list(self._tasks))
            except asyncio.exceptions.CancelledError as e:
                if not asyncio.current_task().cancelling():
                    pass
                else:
                    # This CancelledError was due to a timeout
                    raise asyncio.TimeoutError(
                        f"Operation timed out after {self._timeout} seconds"
                    ) from e

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
            if not self._disable_validation:
                self._validate()

            self._events = []
            if not self._tasks:
                self._start(stepwise=True)

            # run the first step
            self.send_event(StartEvent(kwargs))

        # let all steps start
        for name in self._queues:
            self._step_flags[name].set()

        # if we're done, return the result
        if self.is_done:
            return self._retval

        return None

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return len(self._tasks) == 0

    def get_result(self) -> str:
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
        self._retval = ev.msg or None

    def _validate(self) -> None:
        """Validate the workflow to ensure it's well-formed."""
        produced_events = {StartEvent}
        consumed_events = set()

        for name, step_func in get_steps_from_class(self).items():
            step_config: StepConfig = getattr(step_func, "__step_config", None)
            if not step_config:
                raise ValueError(f"Step {name} is missing `@step()` decorator.")

            for event_types in step_config.accepted_events.values():
                for event_type in event_types:
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

    def draw_all_possible_flows(
        self,
        filename: str = "workflow_all_flows.html",
        notebook: bool = False,
    ) -> None:
        """Draws all possible flows of the workflow."""
        from pyvis.network import Network

        net = Network(directed=True, height="750px", width="100%")

        # Add the nodes + edge for stop events
        net.add_node(
            StopEvent.__name__,
            label=StopEvent.__name__,
            color="#FFA07A",
            shape="ellipse",
        )
        net.add_node("_done", label="_done", color="#ADD8E6", shape="box")
        net.add_edge(StopEvent.__name__, "_done")

        # Add nodes from all steps
        for step_name, step_func in get_steps_from_class(self).items():
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )
            if step_config is None:
                continue

            net.add_node(
                step_name, label=step_name, color="#ADD8E6", shape="box"
            )  # Light blue for steps

            all_event_types = {
                event_type
                for event_types in step_config.accepted_events.values()
                for event_type in event_types
            }
            for event_type in all_event_types:
                net.add_node(
                    event_type.__name__,
                    label=event_type.__name__,
                    color="#90EE90",
                    shape="ellipse",
                )  # Light green for events

        # Add edges from all steps
        for step_name, step_func in get_steps_from_class(self).items():
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )

            if step_config is None:
                continue

            for return_type in step_config.return_types:
                if return_type != type(None):
                    net.add_edge(step_name, return_type.__name__)

            for event_types in step_config.accepted_events.values():
                for event_type in event_types:
                    net.add_edge(event_type.__name__, step_name)

        net.show(filename, notebook=notebook)

    def draw_most_recent_execution(
        self,
        filename: str = "workflow_recent_execution.html",
        notebook: bool = False,
    ) -> None:
        """Draws the most recent execution of the workflow."""
        from pyvis.network import Network

        net = Network(directed=True, height="750px", width="100%")

        # Add nodes and edges based on execution history
        for i, (step, events) in enumerate(self._events):
            for event in events:
                event_node = f"{event}_{i}"
                step_node = f"{step}_{i}"
                net.add_node(
                    event_node, label=event, color="#90EE90", shape="ellipse"
                )  # Light green for events
                net.add_node(
                    step_node, label=step, color="#ADD8E6", shape="box"
                )  # Light blue for steps
                net.add_edge(event_node, step_node)

                if i > 0:
                    prev_step_node = f"{self._events[i - 1][0]}_{i - 1}"
                    net.add_edge(prev_step_node, event_node)

        net.show(filename, notebook=notebook)
