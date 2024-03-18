"""Query plan tool."""

from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.tools.types import BaseTool, ToolMetadata, ToolOutput
from llama_index.core.utils import print_text

DEFAULT_NAME = "query_plan_tool"

QUERYNODE_QUERY_STR_DESC = """\
Question we are asking. This is the query string that will be executed. \
"""

QUERYNODE_TOOL_NAME_DESC = """\
Name of the tool to execute the `query_str`. \
Should NOT be specified if there are subquestions to be specified, in which \
case child_nodes should be nonempty instead.\
"""

QUERYNODE_DEPENDENCIES_DESC = """\
List of sub-questions that need to be answered in order \
to answer the question given by `query_str`.\
Should be blank if there are no sub-questions to be specified, in which case \
`tool_name` is specified.\
"""


class QueryNode(BaseModel):
    """Query node.

    A query node represents a query (query_str) that must be answered.
    It can either be answered by a tool (tool_name), or by a list of child nodes
    (child_nodes).
    The tool_name and child_nodes fields are mutually exclusive.

    """

    # NOTE: inspired from https://github.com/jxnl/openai_function_call/pull/3/files

    id: int = Field(..., description="ID of the query node.")
    query_str: str = Field(..., description=QUERYNODE_QUERY_STR_DESC)
    tool_name: Optional[str] = Field(
        default=None, description="Name of the tool to execute the `query_str`."
    )
    dependencies: List[int] = Field(
        default_factory=list, description=QUERYNODE_DEPENDENCIES_DESC
    )


class QueryPlan(BaseModel):
    """Query plan.

    Contains a list of QueryNode objects (which is a recursive object).
    Out of the list of QueryNode objects, one of them must be the root node.
    The root node is the one that isn't a dependency of any other node.

    """

    nodes: List[QueryNode] = Field(
        ...,
        description="The original question we are asking.",
    )


DEFAULT_DESCRIPTION_PREFIX = """\
This is a query plan tool that takes in a list of tools and executes a \
query plan over these tools to answer a query. The query plan is a DAG of query nodes.

Given a list of tool names and the query plan schema, you \
can choose to generate a query plan to answer a question.

The tool names and descriptions are as follows:
"""


class QueryPlanTool(BaseTool):
    """Query plan tool.

    A tool that takes in a list of tools and executes a query plan.

    """

    def __init__(
        self,
        query_engine_tools: List[BaseTool],
        response_synthesizer: BaseSynthesizer,
        name: str,
        description_prefix: str,
    ) -> None:
        """Initialize."""
        self._query_tools_dict = {t.metadata.name: t for t in query_engine_tools}
        self._response_synthesizer = response_synthesizer
        self._name = name
        self._description_prefix = description_prefix

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: List[BaseTool],
        response_synthesizer: Optional[BaseSynthesizer] = None,
        name: Optional[str] = None,
        description_prefix: Optional[str] = None,
    ) -> "QueryPlanTool":
        """Initialize from defaults."""
        name = name or DEFAULT_NAME
        description_prefix = description_prefix or DEFAULT_DESCRIPTION_PREFIX
        response_synthesizer = response_synthesizer or get_response_synthesizer()

        return cls(
            query_engine_tools=query_engine_tools,
            response_synthesizer=response_synthesizer,
            name=name,
            description_prefix=description_prefix,
        )

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        tools_description = "\n\n".join(
            [
                f"Tool Name: {tool.metadata.name}\n"
                + f"Tool Description: {tool.metadata.description} "
                for tool in self._query_tools_dict.values()
            ]
        )
        # TODO: fill in description with query engine tools.
        description = f"""\
        {self._description_prefix}\n\n
        {tools_description}
        """
        return ToolMetadata(description, self._name, fn_schema=QueryPlan)

    def _execute_node(
        self, node: QueryNode, nodes_dict: Dict[int, QueryNode]
    ) -> ToolOutput:
        """Execute node."""
        print_text(f"Executing node {node.json()}\n", color="blue")
        if len(node.dependencies) > 0:
            print_text(
                f"Executing {len(node.dependencies)} child nodes\n", color="pink"
            )
            child_query_nodes: List[QueryNode] = [
                nodes_dict[dep] for dep in node.dependencies
            ]
            # execute the child nodes first
            child_responses: List[ToolOutput] = [
                self._execute_node(child, nodes_dict) for child in child_query_nodes
            ]
            # form the child Node/NodeWithScore objects
            child_nodes = []
            for child_query_node, child_response in zip(
                child_query_nodes, child_responses
            ):
                node_text = (
                    f"Query: {child_query_node.query_str}\n"
                    f"Response: {child_response!s}\n"
                )
                child_node = TextNode(text=node_text)
                child_nodes.append(child_node)
            # use response synthesizer to combine results
            child_nodes_with_scores = [
                NodeWithScore(node=n, score=1.0) for n in child_nodes
            ]
            response_obj = self._response_synthesizer.synthesize(
                query=node.query_str,
                nodes=child_nodes_with_scores,
            )
            response = ToolOutput(
                content=str(response_obj),
                tool_name=node.query_str,
                raw_input={"query": node.query_str},
                raw_output=response_obj,
            )

        else:
            # this is a leaf request, execute the query string using the specified tool
            tool = self._query_tools_dict[node.tool_name]
            print_text(f"Selected Tool: {tool.metadata}\n", color="pink")
            response = tool(node.query_str)
        print_text(
            "Executed query, got response.\n"
            f"Query: {node.query_str}\n"
            f"Response: {response!s}\n",
            color="blue",
        )
        return response

    def _find_root_nodes(self, nodes_dict: Dict[int, QueryNode]) -> List[QueryNode]:
        """Find root node."""
        # the root node is the one that isn't a dependency of any other node
        node_counts = {node_id: 0 for node_id in nodes_dict}
        for node in nodes_dict.values():
            for dep in node.dependencies:
                node_counts[dep] += 1
        root_node_ids = [
            node_id for node_id, count in node_counts.items() if count == 0
        ]
        return [nodes_dict[node_id] for node_id in root_node_ids]

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        # the kwargs represented as a JSON object
        # should be a QueryPlan object
        query_plan = QueryPlan(**kwargs)

        nodes_dict = {node.id: node for node in query_plan.nodes}
        root_nodes = self._find_root_nodes(nodes_dict)
        if len(root_nodes) > 1:
            raise ValueError("Query plan should have exactly one root node.")

        return self._execute_node(root_nodes[0], nodes_dict)
