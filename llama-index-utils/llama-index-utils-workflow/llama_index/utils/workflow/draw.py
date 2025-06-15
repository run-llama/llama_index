from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class DrawWorkflowNode:
    """Represents a node in the workflow graph."""

    id: str
    label: str
    node_type: str  # 'step', 'event', 'external'
    title: Optional[str] = None
    event_type: Optional[type] = (
        None  # Store the actual event type for styling decisions
    )


@dataclass
class DrawWorkflowEdge:
    """Represents an edge in the workflow graph."""

    source: str
    target: str


@dataclass
class DrawWorkflowGraph:
    """Intermediate representation of workflow structure."""

    nodes: List[DrawWorkflowNode]
    edges: List[DrawWorkflowEdge]


def _truncate_label(label: str, max_length: int) -> str:
    """Helper to truncate long labels."""
    return label if len(label) <= max_length else f"{label[: max_length - 1]}*"


def _extract_workflow_structure(
    workflow: Workflow, max_label_length: Optional[int] = None
) -> DrawWorkflowGraph:
    """Extract workflow structure into an intermediate representation."""
    # Get workflow steps
    steps = get_steps_from_class(workflow)
    if not steps:
        steps = get_steps_from_instance(workflow)

    nodes = []
    edges = []
    added_nodes = set()  # Track added node IDs to avoid duplicates

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

    # First pass: Add all nodes
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        # Add step node
        step_label = (
            _truncate_label(step_name, max_label_length)
            if max_label_length
            else step_name
        )
        step_title = (
            step_name
            if max_label_length and len(step_name) > max_label_length
            else None
        )

        if step_name not in added_nodes:
            nodes.append(
                DrawWorkflowNode(
                    id=step_name,
                    label=step_label,
                    node_type="step",
                    title=step_title,
                )
            )
            added_nodes.add(step_name)

        # Add event nodes for accepted events
        for event_type in step_config.accepted_events:
            if event_type == StopEvent and event_type != current_stop_event:
                continue

            event_label = (
                _truncate_label(event_type.__name__, max_label_length)
                if max_label_length
                else event_type.__name__
            )
            event_title = (
                event_type.__name__
                if max_label_length and len(event_type.__name__) > max_label_length
                else None
            )

            if event_type.__name__ not in added_nodes:
                nodes.append(
                    DrawWorkflowNode(
                        id=event_type.__name__,
                        label=event_label,
                        node_type="event",
                        title=event_title,
                        event_type=event_type,
                    )
                )
                added_nodes.add(event_type.__name__)

        # Add event nodes for return types
        for return_type in step_config.return_types:
            if return_type is type(None):
                continue

            return_label = (
                _truncate_label(return_type.__name__, max_label_length)
                if max_label_length
                else return_type.__name__
            )
            return_title = (
                return_type.__name__
                if max_label_length and len(return_type.__name__) > max_label_length
                else None
            )

            if return_type.__name__ not in added_nodes:
                nodes.append(
                    DrawWorkflowNode(
                        id=return_type.__name__,
                        label=return_label,
                        node_type="event",
                        title=return_title,
                        event_type=return_type,
                    )
                )
                added_nodes.add(return_type.__name__)

            # Add external_step node when InputRequiredEvent is found
            if (
                issubclass(return_type, InputRequiredEvent)
                and "external_step" not in added_nodes
            ):
                nodes.append(
                    DrawWorkflowNode(
                        id="external_step",
                        label="external_step",
                        node_type="external",
                    )
                )
                added_nodes.add("external_step")

    # Second pass: Add edges
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        # Edges from steps to return types
        for return_type in step_config.return_types:
            if return_type is not type(None):
                edges.append(DrawWorkflowEdge(step_name, return_type.__name__))

            if issubclass(return_type, InputRequiredEvent):
                edges.append(DrawWorkflowEdge(return_type.__name__, "external_step"))

        # Edges from events to steps
        for event_type in step_config.accepted_events:
            if step_name == "_done" and issubclass(event_type, StopEvent):
                if current_stop_event:
                    edges.append(
                        DrawWorkflowEdge(current_stop_event.__name__, step_name)
                    )
            else:
                edges.append(DrawWorkflowEdge(event_type.__name__, step_name))

            if issubclass(event_type, HumanResponseEvent):
                edges.append(DrawWorkflowEdge("external_step", event_type.__name__))

    return DrawWorkflowGraph(nodes=nodes, edges=edges)


def _get_node_color(node: DrawWorkflowNode) -> str:
    """Determine color for a node based on its type and event_type."""
    if node.node_type == "step":
        return "#ADD8E6"  # Light blue for steps
    elif node.node_type == "external":
        return "#BEDAE4"  # Light blue-gray for external
    elif node.node_type == "event" and node.event_type:
        return determine_event_color(node.event_type)  # Uses original function
    else:
        return "#90EE90"  # Default light green


def _get_node_shape(node: DrawWorkflowNode) -> str:
    """Determine shape for a node based on its type."""
    if node.node_type == "step" or node.node_type == "external":
        return "box"  # Steps and external_step use box
    elif node.node_type == "event":
        return "ellipse"  # Events use ellipse
    else:
        return "box"  # Default shape


def _render_pyvis(
    graph: DrawWorkflowGraph, filename: str, notebook: bool = False
) -> None:
    """Render workflow graph using Pyvis."""
    from pyvis.network import Network

    net = Network(directed=True, height="750px", width="100%")

    # Add nodes
    for node in graph.nodes:
        color = _get_node_color(node)
        shape = _get_node_shape(node)
        net.add_node(
            node.id,
            label=node.label,
            title=node.title,
            color=color,
            shape=shape,
        )

    # Add edges
    for edge in graph.edges:
        net.add_edge(edge.source, edge.target)

    net.show(filename, notebook=notebook)


def _clean_id_for_mermaid(name: str) -> str:
    """Convert a name to a valid Mermaid ID."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def _get_mermaid_css_class(node: DrawWorkflowNode) -> str:
    """Determine CSS class for a node in Mermaid based on its type and event_type."""
    if node.node_type == "step":
        return "stepStyle"
    elif node.node_type == "external":
        return "externalStyle"
    elif node.node_type == "event" and node.event_type:
        if issubclass(node.event_type, StartEvent):
            return "startEventStyle"
        elif issubclass(node.event_type, StopEvent):
            return "stopEventStyle"
        else:
            return "defaultEventStyle"
    else:
        return "defaultEventStyle"


def _render_mermaid(graph: DrawWorkflowGraph, filename: str) -> str:
    """Render workflow graph using Mermaid."""
    mermaid_lines = ["flowchart TD"]
    added_nodes = set()
    added_edges = set()

    # Add nodes
    for node in graph.nodes:
        # Clean ID for Mermaid
        if node.node_type == "step":
            clean_id = f"step_{_clean_id_for_mermaid(node.id)}"
        elif node.node_type == "external":
            clean_id = node.id  # external_step is already clean
        else:  # event
            clean_id = f"event_{_clean_id_for_mermaid(node.id)}"

        if clean_id not in added_nodes:
            added_nodes.add(clean_id)

            # Format node based on shape
            shape = _get_node_shape(node)
            if shape == "box":
                shape_start, shape_end = "[", "]"
            elif shape == "ellipse":
                shape_start, shape_end = "([", "])"
            else:
                shape_start, shape_end = "[", "]"

            css_class = _get_mermaid_css_class(node)
            mermaid_lines.append(
                f'    {clean_id}{shape_start}"{node.label}"{shape_end}:::{css_class}'
            )

    # Add edges
    for edge in graph.edges:
        source_node = next(n for n in graph.nodes if n.id == edge.source)
        target_node = next(n for n in graph.nodes if n.id == edge.target)

        if source_node.node_type == "step":
            source_id = f"step_{_clean_id_for_mermaid(edge.source)}"
        elif source_node.node_type == "external":
            source_id = edge.source
        else:  # event
            source_id = f"event_{_clean_id_for_mermaid(edge.source)}"

        if target_node.node_type == "step":
            target_id = f"step_{_clean_id_for_mermaid(edge.target)}"
        elif target_node.node_type == "external":
            target_id = edge.target
        else:  # event
            target_id = f"event_{_clean_id_for_mermaid(edge.target)}"

        edge_str = f"{source_id} --> {target_id}"
        if edge_str not in added_edges:
            added_edges.add(edge_str)
            mermaid_lines.append(f"    {edge_str}")

    # Add style definitions
    mermaid_lines.extend(
        [
            "    classDef stepStyle fill:#ADD8E6,color:#000000,line-height:1.2",
            "    classDef externalStyle fill:#BEDAE4,color:#000000,line-height:1.2",
            "    classDef startEventStyle fill:#E27AFF,color:#000000",
            "    classDef stopEventStyle fill:#FFA07A,color:#000000",
            "    classDef defaultEventStyle fill:#90EE90,color:#000000",
        ]
    )

    diagram_string = "\n".join(mermaid_lines)

    if filename:
        with open(filename, "w") as f:
            f.write(diagram_string)

    return diagram_string


def draw_all_possible_flows(
    workflow: Workflow,
    filename: str = "workflow_all_flows.html",
    notebook: bool = False,
    max_label_length: Optional[int] = None,
) -> None:
    """
    Draws all possible flows of the workflow using Pyvis.

    Args:
        workflow: The workflow to visualize
        filename: Output HTML filename
        notebook: Whether running in notebook environment
        max_label_length: Maximum label length before truncation (None = no limit)

    """
    graph = _extract_workflow_structure(workflow, max_label_length)
    _render_pyvis(graph, filename, notebook)


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


def determine_event_color(event_type):
    """Determine color for an event type."""
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


def draw_all_possible_flows_mermaid(
    workflow: Workflow,
    filename: str = "workflow_all_flows.mermaid",
    max_label_length: Optional[int] = None,
) -> str:
    """
    Draws all possible flows of the workflow as a Mermaid diagram.

    Args:
        workflow: The workflow to visualize
        filename: Output Mermaid filename
        max_label_length: Maximum label length before truncation (None = no limit)

    Returns:
        The Mermaid diagram as a string

    """
    graph = _extract_workflow_structure(workflow, max_label_length)
    return _render_mermaid(graph, filename)
