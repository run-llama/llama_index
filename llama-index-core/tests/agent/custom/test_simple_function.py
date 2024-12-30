"""Test simple function agent."""

from typing import Any, Dict, Tuple
import pytest
from llama_index.core.agent.custom.simple_function import FnAgentWorker


def mock_foo_fn_no_state_param() -> Tuple[None, bool]:
    """Mock agent input function without a state."""
    return None, True


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


async def async_mock_foo_fn(state: dict) -> Tuple[Dict[str, Any], bool]:
    """Mock async agent input function."""
    return mock_foo_fn(state)


def test_fn_agent() -> None:
    """Test function agent."""
    agent = FnAgentWorker(fn=mock_foo_fn, initial_state={"max_count": 5}).as_agent()
    response = agent.query("hello")
    assert str(response) == "hello:foo:foo:foo:foo:foo"

    with pytest.raises(ValueError):
        agent = FnAgentWorker(fn=mock_foo_fn).as_agent()
        response = agent.query("hello")


def test_fn_agent_init() -> None:
    """Test function agent init."""
    with pytest.raises(ValueError) as error_info:
        FnAgentWorker(fn=mock_foo_fn_no_state_param).as_agent()

    assert (
        str(error_info.value)
        == "StatefulFnComponent must have 'state' as required parameters"
    )


@pytest.mark.asyncio()
async def test_run_async_step() -> None:
    """Test run async step."""
    agent_without_async_fn = FnAgentWorker(
        fn=mock_foo_fn, asnc_fn=None, initial_state={"max_count": 5}
    ).as_agent()

    response = await agent_without_async_fn.aquery("hello")
    assert str(response) == "hello:foo:foo:foo:foo:foo"

    agent_with_async_fn = FnAgentWorker(
        fn=mock_foo_fn, async_fn=async_mock_foo_fn, initial_state={"max_count": 5}
    ).as_agent()

    response = await agent_with_async_fn.aquery("hello")
    assert str(response) == "hello:foo:foo:foo:foo:foo"
