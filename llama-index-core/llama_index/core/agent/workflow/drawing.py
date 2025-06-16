import uuid

from typing import List, Union, Dict
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
    CodeActAgent,
    BaseWorkflowAgent,
)


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
    tools = agent.tools
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
    tools = agent.tools
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
