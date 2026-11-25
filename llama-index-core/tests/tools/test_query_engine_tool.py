"""Test tools."""

from typing import Type, cast

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool


class MockQueryEngine(CustomQueryEngine):
    """Custom query engine."""

    def custom_query(self, query_str: str) -> str:
        """Query."""
        return "custom_" + query_str


def test_query_engine_tool() -> None:
    """Test query engine tool."""
    query_engine = MockQueryEngine()  # type: ignore[call-arg]

    query_tool = QueryEngineTool.from_defaults(query_engine)

    # make sure both input formats work given function schema that assumes defaults
    response = query_tool("hello world")
    assert str(response) == "custom_hello world"
    response = query_tool(input="foo")
    assert str(response) == "custom_foo"

    fn_schema_cls = cast(Type[BaseModel], query_tool.metadata.fn_schema)
    fn_schema_obj = cast(BaseModel, fn_schema_cls(input="bar"))
    response = query_tool(**fn_schema_obj.model_dump())
    assert str(response) == "custom_bar"

    # test resolve input errors
    query_tool = QueryEngineTool.from_defaults(query_engine)
    response = query_tool(tmp="hello world")
    assert str(response) == "custom_{'tmp': 'hello world'}"

    with pytest.raises(ValueError):
        query_tool = QueryEngineTool.from_defaults(
            query_engine, resolve_input_errors=False
        )
        response = query_tool(tmp="hello world")


def test_query_plan_tool_dag_memoization_and_cycle_detection() -> None:
    """Test QueryPlanTool executes shared DAG dependency nodes once and detects cycles."""
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.tools.query_plan import QueryNode, QueryPlanTool
    from llama_index.core.tools.types import ToolMetadata

    call_counts: dict[str, int] = {}

    class CountingQueryEngine(CustomQueryEngine):
        def custom_query(self, query_str: str) -> str:
            call_counts[query_str] = call_counts.get(query_str, 0) + 1
            return f"result({query_str})"

    leaf_tool = QueryEngineTool(
        query_engine=CountingQueryEngine(),  # type: ignore[call-arg]
        metadata=ToolMetadata(name="leaf_tool", description="Leaf tool"),
    )
    synthesizer = get_response_synthesizer(llm=MockLLM())
    plan_tool = QueryPlanTool.from_defaults(
        query_engine_tools=[leaf_tool],
        response_synthesizer=synthesizer,
    )

    # Diamond DAG: root (4) -> [2, 3], and both 2 and 3 depend on shared leaf (1).
    output = plan_tool(
        nodes=[
            QueryNode(id=1, query_str="q1", tool_name="leaf_tool", dependencies=[]),
            QueryNode(id=2, query_str="q2", dependencies=[1]),
            QueryNode(id=3, query_str="q3", dependencies=[1]),
            QueryNode(id=4, query_str="q4", dependencies=[2, 3]),
        ]
    )
    assert call_counts["q1"] == 1
    assert output is not None

    # Root (1) -> 2 -> 3 -> 2 cycle should raise ValueError rather than RecursionError.
    with pytest.raises(ValueError, match="Cycle detected in query plan"):
        plan_tool(
            nodes=[
                QueryNode(id=1, query_str="root", dependencies=[2]),
                QueryNode(id=2, query_str="a", dependencies=[3]),
                QueryNode(id=3, query_str="b", dependencies=[2]),
            ]
        )
