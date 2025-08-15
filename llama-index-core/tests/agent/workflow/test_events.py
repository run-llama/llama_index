import pytest
import json

from typing import Tuple
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.bridge.pydantic import BaseModel, ValidationError
from llama_index.core.agent.workflow.workflow_events import (
    AgentWorkflowStartEvent,
    AgentOutput,
    PydanticConversionWarning,
    AgentStreamStructuredOutput,
)
from llama_index.core.memory import Memory


@pytest.fixture()
def example_agent_output() -> dict:
    return {
        "response": ChatMessage(role="user", content="30 times 2 is 60."),
        "tool_calls": [
            ToolSelection(
                tool_id="1", tool_name="multiply", tool_kwargs={"i": 30, "j": 2}
            )
        ],
        "raw": '{"role": "user", "content": "30 times 2 is 60."}',
        "structured_response": {"operation": "30 times 2", "result": 60},
        "current_agent_name": "CalculatorAgent",
    }


@pytest.fixture()
def example_agent_stream_structured_output() -> Tuple[dict, str]:
    d = {"output": {"flavor": "strawberry", "extra_sugar": False}}
    return d, json.dumps(d["output"], indent=4)


class MathResult(BaseModel):
    operation: str
    result: int


class WrongMathResult(BaseModel):
    operation: str
    result: str


class Flavor(BaseModel):
    flavor: str
    extra_sugar: bool


def test_agent_workflow_start_event():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[ChatMessage(role="user", content="Hello, world!")],
        max_iterations=10,
    )
    assert event.user_msg == "Hello, world!"
    assert event.chat_history[0].role.value == "user"
    assert event.chat_history[0].content == "Hello, world!"
    assert event.max_iterations == 10


def test_agent_workflow_start_event_with_dict():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[{"role": "user", "content": "Hello, world!"}],
        max_iterations=10,
    )
    assert event.user_msg == "Hello, world!"
    assert event.chat_history[0].role.value == "user"
    assert event.chat_history[0].content == "Hello, world!"
    assert event.max_iterations == 10


def test_agent_workflow_start_event_to_dict():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[ChatMessage(role="user", content="Hello, world!")],
        max_iterations=10,
        memory=Memory.from_defaults(),
    )

    # Memory is not included in the dump
    dump = event.model_dump()
    assert len(dump) == 3
    assert dump["user_msg"] == "Hello, world!"
    assert dump["chat_history"][0]["role"] == "user"
    assert dump["chat_history"][0]["blocks"][0]["text"] == "Hello, world!"
    assert dump["max_iterations"] == 10


def test_agent_output_with_structured_response(example_agent_output: dict) -> None:
    try:
        agent_output = AgentOutput.model_validate(example_agent_output)
        success = True
    except ValidationError:
        success = False
    assert success
    assert agent_output.get_pydantic_model(MathResult) == MathResult.model_validate(
        example_agent_output["structured_response"]
    )
    with pytest.warns(PydanticConversionWarning):
        a = agent_output.get_pydantic_model(WrongMathResult)
    assert a is None


def test_agent_stream_structured_output(
    example_agent_stream_structured_output: Tuple[dict, str],
):
    try:
        ev = AgentStreamStructuredOutput.model_validate(
            example_agent_stream_structured_output[0]
        )
        success = True
    except ValidationError:
        success = False
    assert success
    assert str(ev) == example_agent_stream_structured_output[1]
    assert ev.get_pydantic_model(Flavor) == Flavor(
        flavor="strawberry", extra_sugar=False
    )
    with pytest.warns(PydanticConversionWarning):
        b = ev.get_pydantic_model(MathResult)
    assert b is None
