"""Test agent executor."""

import uuid
from typing import Any

from llama_index.agent.executor.base import AgentEngine
from llama_index.agent.types import BaseAgentStepEngine, Task, TaskStep, TaskStepOutput
from llama_index.chat_engine.types import AgentChatResponse


# define mock step engine
class MockAgentStepEngine(BaseAgentStepEngine):
    """Mock agent step engine."""

    def __init__(self, limit: int = 2):
        """Initialize."""
        self.limit = limit

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        counter = 0
        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            memory=task.memory,
            step_state={"counter": counter},
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        counter = step.step_state["counter"] + 1
        step.step_state["counter"] = counter
        is_done = counter >= self.limit

        new_steps = [step.get_next_step(step_id=str(uuid.uuid4()))]

        return TaskStepOutput(
            output=AgentChatResponse(response=f"counter: {counter}"),
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return self.run_step(step=step, task=task, **kwargs)

    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        raise NotImplementedError

    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError


def test_agent() -> None:
    """Test executor."""
    agent_engine = AgentEngine(step_executor=MockAgentStepEngine(limit=2))

    # test create_task
    task = agent_engine.create_task("hello world")
    assert task.input == "hello world"
    assert task.task_id in agent_engine.state.task_dict

    # test run step
    step_output = agent_engine.run_step(task=task)
    assert step_output.task_step.step_state["counter"] == 1
    assert str(step_output.output) == "counter: 1"
    assert step_output.is_last is False

    # test run step again
    step_output = agent_engine.run_step(task=task)
    assert step_output.task_step.step_state["counter"] == 2
    assert str(step_output.output) == "counter: 2"
    assert step_output.is_last is True

    # test e2e chat
    # NOTE: to use chat, output needs to be AgentChatResponse
    agent_engine = AgentEngine(step_executor=MockAgentStepEngine(limit=10))
    response = agent_engine.chat("hello world")
    assert str(response) == "counter: 10"
    assert agent_engine.state.task_dict == {}
