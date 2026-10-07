import pytest

from llama_index.core.tools.query_plan import QueryNode, QueryPlanTool
from llama_index.core.tools.types import ToolOutput


class _DummySynthesizer:
    def synthesize(self, **kwargs):
        return "ok"


class _DummyTool:
    def __init__(self, name: str) -> None:
        self.metadata = type("Meta", (), {"name": name, "description": name})()

    def __call__(self, query_str: str) -> ToolOutput:
        return ToolOutput(
            content=query_str,
            tool_name=self.metadata.name,
            raw_input={"query": query_str},
            raw_output=query_str,
        )


def _tool() -> QueryPlanTool:
    return QueryPlanTool(
        query_engine_tools=[_DummyTool("search")],
        response_synthesizer=_DummySynthesizer(),
        name="query_plan_tool",
        description_prefix="plan",
    )


def test_query_plan_empty_nodes_raises() -> None:
    with pytest.raises(ValueError, match="exactly one root node"):
        _tool()(nodes=[])


def test_query_plan_dangling_dependency_does_not_keyerror() -> None:
    plan_tool = _tool()
    nodes_dict = {
        1: QueryNode(id=1, query_str="q", tool_name="search", dependencies=[99]),
    }
    roots = plan_tool._find_root_nodes(nodes_dict)
    assert [node.id for node in roots] == [1]


def test_query_plan_missing_dependency_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing dependencies"):
        _tool()(
            nodes=[
                QueryNode(id=1, query_str="q", tool_name="search", dependencies=[99]),
            ]
        )


def test_query_plan_unknown_tool_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        _tool()(
            nodes=[
                QueryNode(id=1, query_str="q", tool_name="missing", dependencies=[]),
            ]
        )


def test_query_plan_single_root_executes() -> None:
    output = _tool()(
        nodes=[
            QueryNode(id=1, query_str="hello", tool_name="search", dependencies=[]),
        ]
    )
    assert output.content == "hello"
