"""Test agent executor."""

import uuid
from typing import Any

from llama_index.agent.runner.base import AgentRunner
from llama_index.agent.runner.parallel import ParallelAgentRunner
from llama_index.agent.types import BaseAgentWorker, Task, TaskStep, TaskStepOutput
from llama_index.chat_engine.types import AgentChatResponse


# define mock agent worker
class MockAgentWorker(BaseAgentWorker):
    """Mock agent agent worker."""

    def __init__(self, limit: int = 2):
        """Initialize."""
        self.limit = limit

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        counter = 0
        task.extra_state["counter"] = counter
        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            memory=task.memory,
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        counter = task.extra_state["counter"] + 1
        task.extra_state["counter"] = counter
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

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""


# define mock agent worker
class MockForkStepEngine(BaseAgentWorker):
    """Mock agent worker that adds an exponential # steps."""

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
            step_state={"num": "0", "counter": counter},
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        counter = step.step_state["counter"] + 1
        step.step_state["counter"] = counter
        is_done = counter >= self.limit

        cur_num = step.step_state["num"]

        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    step_state={"num": cur_num + "0", "counter": counter},
                ),
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    step_state={"num": cur_num + "1", "counter": counter},
                ),
            ]

        return TaskStepOutput(
            output=AgentChatResponse(response=cur_num),
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

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""


def test_agent() -> None:
    """Test executor."""
    agent_runner = AgentRunner(agent_worker=MockAgentWorker(limit=2))

    # test create_task
    task = agent_runner.create_task("hello world")
    assert task.input == "hello world"
    assert task.task_id in agent_runner.state.task_dict

    # test run step
    step_output = agent_runner.run_step(task.task_id)
    assert task.extra_state["counter"] == 1
    assert str(step_output.output) == "counter: 1"
    assert step_output.is_last is False

    # test list task, get task
    assert len(agent_runner.list_tasks()) == 1
    assert agent_runner.get_task(task_id=task.task_id) == task

    # test run step again
    step_output = agent_runner.run_step(task.task_id)
    assert task.extra_state["counter"] == 2
    assert str(step_output.output) == "counter: 2"
    assert step_output.is_last is True
    assert len(agent_runner.state.task_dict[task.task_id].completed_steps) == 2

    # test e2e chat
    # NOTE: to use chat, output needs to be AgentChatResponse
    agent_runner = AgentRunner(agent_worker=MockAgentWorker(limit=10))
    response = agent_runner.chat("hello world")
    assert str(response) == "counter: 10"
    assert len(agent_runner.state.task_dict) == 1


def test_dag_agent() -> None:
    """Test DAG agent executor."""
    agent_runner = ParallelAgentRunner(agent_worker=MockForkStepEngine(limit=2))

    # test create_task
    task = agent_runner.create_task("hello world")

    # test run step
    step_outputs = agent_runner.run_steps_in_queue(task_id=task.task_id)
    step_output = step_outputs[0]
    assert step_output.task_step.step_state["num"] == "0"
    assert str(step_output.output) == "0"
    assert step_output.is_last is False

    # test run step again
    step_outputs = agent_runner.run_steps_in_queue(task_id=task.task_id)
    assert step_outputs[0].task_step.step_state["num"] == "00"
    assert step_outputs[1].task_step.step_state["num"] == "01"
    # TODO: deal with having multiple `is_last` outputs in chat later.
    assert step_outputs[0].is_last is True
    assert step_outputs[1].is_last is True
    assert len(agent_runner.state.task_dict[task.task_id].completed_steps) == 3


def test_agent_from_llm() -> None:
    from llama_index.agent import OpenAIAgent, ReActAgent
    from llama_index.llms.mock import MockLLM
    from llama_index.llms.openai import OpenAI

    llm = OpenAI()
    agent_runner = AgentRunner.from_llm(llm=llm)
    assert isinstance(agent_runner, OpenAIAgent)
    llm = MockLLM()
    agent_runner = AgentRunner.from_llm(llm=llm)
    assert isinstance(agent_runner, ReActAgent)
