"""Test simple function agent."""

from typing import Any, Dict, Set, Tuple

from llama_index.core.agent.custom.pipeline_worker import (
    QueryPipelineAgentWorker,
)
from llama_index.core.agent.custom.simple_function import FnAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import Task
from llama_index.core.bridge.pydantic import Field
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.query_pipeline import FnComponent, QueryPipeline
from llama_index.core.query_pipeline.components.agent import (
    AgentFnComponent,
    AgentInputComponent,
    CustomAgentComponent,
)
from llama_index.core.query_pipeline.components.stateful import StatefulFnComponent
import pytest

def mock_foo_fn(state: dict) -> Tuple[Dict[str, Any], bool]:
    """Mock agent input function."""

    if "max_count" not in state:
        raise ValueError("max_count must be specified.")

    if "input" not in state:
        state["input"] = state["__task__"].input
        state["count"] = 0
        is_done = False
    else:
        state["input"] = state["input"] + ":foo"
        state["count"] += 1
        is_done = state["count"] >= state["max_count"]

    state["__output__"] = state["input"]

    return state, is_done


def test_fn_agent() -> None:
    """Test function agent."""

    agent = FnAgentWorker(fn=mock_foo_fn, initial_state={"max_count": 5}).as_agent()
    response = agent.query("hello")
    assert str(response) == "hello:foo:foo:foo:foo:foo"

    with pytest.raises(ValueError):
        agent = FnAgentWorker(fn=mock_foo_fn).as_agent()
        response = agent.query("hello")
    
    