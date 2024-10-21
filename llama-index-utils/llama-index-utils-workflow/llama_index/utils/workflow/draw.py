from typing import Optional

from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
)
from llama_index.core.workflow.decorators import StepConfig
from llama_index.core.workflow.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)
from llama_index.core.workflow.workflow import Workflow


def draw_all_possible_flows(
    workflow: Workflow,
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

    # Add nodes from all steps
    steps = get_steps_from_class(workflow)
    if not steps:
        # If no steps are defined in the class, try to get them from the instance
        steps = get_steps_from_instance(workflow)

    step_config: Optional[StepConfig] = None
    net_has_external_step_node: bool = False
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        net.add_node(
            step_name, label=step_name, color="#ADD8E6", shape="box"
        )  # Light blue for steps

        for event_type in step_config.accepted_events:
            net.add_node(
                event_type.__name__,
                label=event_type.__name__,
                color="#90EE90" if event_type != StartEvent else "#E27AFF",
                shape="ellipse",
            )  # Light green for events

        for return_type in step_config.return_types:
            print(f"{return_type}", flush=True)
            if return_type == type(None):
                continue

            if issubclass(return_type, InputRequiredEvent):
                net.add_node(
                    return_type.__name__,
                    label=return_type.__name__,
                    color="#90EE90",
                    shape="ellipse",
                )  # Light green for events

                if not net_has_external_step_node:
                    # add node for conceptual external step
                    net.add_node(
                        f"external_step",
                        label="external_step",
                        color="#BEDAE4",
                        shape="box",
                    )

    # Add edges from all steps
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)

        if step_config is None:
            continue

        for return_type in step_config.return_types:
            if return_type != type(None):
                net.add_edge(step_name, return_type.__name__)

            if issubclass(return_type, InputRequiredEvent):
                net.add_edge(return_type.__name__, f"external_step")

        for event_type in step_config.accepted_events:
            net.add_edge(event_type.__name__, step_name)

            if issubclass(event_type, HumanResponseEvent):
                net.add_edge(
                    f"external_step",
                    event_type.__name__,
                )

    net.show(filename, notebook=notebook)


def draw_most_recent_execution(
    workflow: Workflow,
    filename: str = "workflow_recent_execution.html",
    notebook: bool = False,
) -> None:
    """Draws the most recent execution of the workflow."""
    from pyvis.network import Network

    net = Network(directed=True, height="750px", width="100%")

    # Add nodes and edges based on execution history
    existing_context = next(iter(workflow._contexts), None)
    if existing_context is None:
        raise ValueError("No runs found in workflow")

    for i, (step, event) in enumerate(existing_context._accepted_events):
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
            prev_step_node = f"{existing_context._accepted_events[i - 1][0]}_{i - 1}"
            net.add_edge(prev_step_node, event_node)

    net.show(filename, notebook=notebook)
