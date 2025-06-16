import uuid

from typing import Optional, List, Union, Dict, cast
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
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
    CodeActAgent,
    BaseWorkflowAgent,
)
from llama_index.core.tools import BaseTool, AsyncBaseTool


def _truncate_label(label: str, max_length: int) -> str:
    """Helper to truncate long labels."""
    return label if len(label) <= max_length else f"{label[: max_length - 1]}*"


def draw_all_possible_flows(
    workflow: Workflow,
    filename: str = "workflow_all_flows.html",
    notebook: bool = False,
    max_label_length: Optional[int] = None,
) -> None:
    """
    Draws all possible flows of the workflow.

    Args:
        workflow: The workflow to visualize
        filename: Output HTML filename
        notebook: Whether running in notebook environment
        max_label_length: Maximum label length before truncation (None = no limit)

    """
    from pyvis.network import Network

    net = Network(directed=True, height="750px", width="100%")

    # Add nodes from all steps
    steps = get_steps_from_class(workflow)
    if not steps:
        # If no steps are defined in the class, try to get them from the instance
        steps = get_steps_from_instance(workflow)

    step_config: Optional[StepConfig] = None

    # Only one kind of `StopEvent` is allowed in a `Workflow`.
    # Assuming that `Workflow` is validated before drawing, it's enough to find the first one.
    current_stop_event = None
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        for return_type in step_config.return_types:
            if issubclass(return_type, StopEvent):
                current_stop_event = return_type
                break

        if current_stop_event:
            break

    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        # Handle label truncation for steps
        if max_label_length is not None:
            step_label = _truncate_label(step_name, max_label_length)
            step_title = step_name if len(step_name) > max_label_length else None
        else:
            step_label = step_name
            step_title = None

        net.add_node(
            step_name, label=step_label, title=step_title, color="#ADD8E6", shape="box"
        )  # Light blue for steps

        for event_type in step_config.accepted_events:
            if event_type == StopEvent and event_type != current_stop_event:
                continue

            # Handle label truncation for events
            if max_label_length is not None:
                event_label = _truncate_label(event_type.__name__, max_label_length)
                event_title = (
                    event_type.__name__
                    if len(event_type.__name__) > max_label_length
                    else None
                )
            else:
                event_label = event_type.__name__
                event_title = None

            net.add_node(
                event_type.__name__,
                label=event_label,
                title=event_title,
                color=determine_event_color(event_type),
                shape="ellipse",
            )

        for return_type in step_config.return_types:
            if return_type is type(None):
                continue

            # Handle label truncation for return type events
            if max_label_length is not None:
                return_label = _truncate_label(return_type.__name__, max_label_length)
                return_title = (
                    return_type.__name__
                    if len(return_type.__name__) > max_label_length
                    else None
                )
            else:
                return_label = return_type.__name__
                return_title = None

            net.add_node(
                return_type.__name__,
                label=return_label,
                title=return_title,
                color=determine_event_color(return_type),
                shape="ellipse",
            )

            if issubclass(return_type, InputRequiredEvent):
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
            if return_type is not type(None):
                net.add_edge(step_name, return_type.__name__)

            if issubclass(return_type, InputRequiredEvent):
                net.add_edge(return_type.__name__, f"external_step")

        for event_type in step_config.accepted_events:
            if step_name == "_done" and issubclass(event_type, StopEvent):
                net.add_edge(current_stop_event.__name__, step_name)
            else:
                net.add_edge(event_type.__name__, step_name)

            if issubclass(event_type, HumanResponseEvent):
                net.add_edge(
                    f"external_step",
                    event_type.__name__,
                )

    net.show(filename, notebook=notebook)


def determine_event_color(event_type):
    if issubclass(event_type, StartEvent):
        # Pink for start events
        event_color = "#E27AFF"
    elif issubclass(event_type, StopEvent):
        # Orange for stop events
        event_color = "#FFA07A"
    else:
        # Light green for other events
        event_color = "#90EE90"
    return event_color


def draw_most_recent_execution(
    workflow: Workflow,
    filename: str = "workflow_recent_execution.html",
    notebook: bool = False,
    max_label_length: Optional[int] = None,
) -> None:
    """
    Draws the most recent execution of the workflow.

    Args:
        workflow: The workflow to visualize
        filename: Output HTML filename
        notebook: Whether running in notebook environment
        max_label_length: Maximum label length before truncation (None = no limit)

    """
    from pyvis.network import Network

    net = Network(directed=True, height="750px", width="100%")

    # Add nodes and edges based on execution history
    existing_context = next(iter(workflow._contexts), None)
    if existing_context is None:
        raise ValueError("No runs found in workflow")

    for i, (step, event) in enumerate(existing_context._accepted_events):
        event_node = f"{event}_{i}"
        step_node = f"{step}_{i}"

        if max_label_length is not None:
            event_label = _truncate_label(event, max_label_length)
            step_label = _truncate_label(step, max_label_length)
        else:
            event_label = event
            step_label = step

        net.add_node(
            event_node, label=event_label, color="#90EE90", shape="ellipse"
        )  # Light green for events
        net.add_node(
            step_node, label=step_label, color="#ADD8E6", shape="box"
        )  # Light blue for steps
        net.add_edge(event_node, step_node)

        if i > 0:
            prev_step_node = f"{existing_context._accepted_events[i - 1][0]}_{i - 1}"
            net.add_edge(prev_step_node, event_node)

    net.show(filename, notebook=notebook)


def draw_agent_with_tools(
    agent: BaseWorkflowAgent,
    filename: str = "agent_with_tools.html",
    notebook: bool = False,
) -> str:
    """
    > **NOTE**: *PyVis is needed for this function*.

    Draw an agent with its tool as a flowchart.

    Args:
        agent (BaseWorkflowAgent): agent workflow.
        filename (str): name of the HTML file to save the flowchart to. Defaults to 'agent_workflow.html'.
        notebook (bool): whether or not this is displayed within a notebook (.ipynb). Defaults to False.

    Returns:
        str: the path to the file where the flowchart was saved.

    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "PyVis is not installed, please make sure to install it by running `pip install pyvis`"
        )
    d = Network(directed=True, height="750px", width="100%")
    node_color = "#90EE90"
    if isinstance(agent, ReActAgent):
        node_color = "#E27AFF"
    elif isinstance(agent, CodeActAgent):
        node_color = "#66ccff"
    d.add_node(n_id="A", label=agent.name, color=node_color, shape="ellipse")
    tools = cast(Union[List[Union[BaseTool, AsyncBaseTool]], None], agent.tools)
    node_names: List[str] = []
    if tools is not None and len(tools) > 0:
        for i in range(len(tools)):
            node_names.append(f"B{i + 1}")
            d.add_node(
                n_id=f"B{i + 1}",
                label=f"Tool {i + 1}: {tools[i].metadata.get_name()}",
                color="#ff9966",
                shape="ellipsis",
            )
    for n in node_names:
        d.add_edge("A", n)
    d.show(name=filename, notebook=notebook)
    return filename


def _get_agent_tool_names(agent: BaseWorkflowAgent) -> Union[List[str], None]:
    tools = cast(Union[List[Union[BaseTool, AsyncBaseTool]], None], agent.tools)
    if tools and len(tools) > 0:
        return [tool.metadata.get_name() for tool in tools]
    return None


def draw_agent_workflow(
    agent_workflow: AgentWorkflow,
    filename: str = "agent_workflow.html",
    notebook: bool = False,
) -> str:
    """
    > **NOTE**: *PyVis is needed for this function*.

    Draw an agent workflow as a flowchart.

    Args:
        agent_workflow (AgentWorkflow): agent workflow.
        filename (str): name of the HTML file to save the flowchart to. Defaults to 'agent_workflow.html'.
        notebook (bool): whether or not this is displayed within a notebook (.ipynb). Defaults to False.

    Returns:
        str: the path to the file where the flowchart was saved.

    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "PyVis is not installed, please make sure to install it by running `pip install pyvis`"
        )
    d = Network(directed=True, height="750px", width="100%")
    d.add_node(n_id="base", label="AgentWorkflow", color="#90EE90", shape="diamond")
    agents = agent_workflow.agents
    can_handoff_to: Dict[str, Union[List[str], None]] = {
        agent: agents[agent].can_handoff_to for agent in agents
    }
    for agent in agents:
        d.add_node(n_id=agent, label=agent, color="#66ccff", shape="ellipsis")
        d.add_edge("base", agent, title="Agent")
        tools = _get_agent_tool_names(agents[agent])
        if tools:
            for toolname in tools:
                d.add_node(
                    n_id=toolname, label=toolname, color="#ff9966", shape="square"
                )
                d.add_edge(agent, toolname, title="Tool")
        handoff_possibilities = can_handoff_to[agent]
        if handoff_possibilities and len(handoff_possibilities) > 0:
            for handoff_possibility in handoff_possibilities:
                node_id = str(uuid.uuid4())
                d.add_node(
                    n_id=node_id,
                    label=handoff_possibility,
                    color="#E27AFF",
                    shape="ellipsis",
                )
                d.add_edge(agent, node_id, title="Can hand off to")
    d.show(name=filename, notebook=notebook)
    return filename
