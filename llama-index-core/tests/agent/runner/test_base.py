"""Test agent executor."""

import uuid
from typing import Any, List, cast
import llama_index.core.instrumentation as instrument
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.runner.parallel import ParallelAgentRunner
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
)
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.tools.types import ToolOutput

dispatcher = instrument.get_dispatcher()


class _TestEventHandler(BaseEventHandler):
    events: List[BaseEvent] = []

    @classmethod
    def class_name(cls):
        return "_TestEventHandler"

    def handle(self, e: BaseEvent):
        self.events.append(e)


# define mock agent worker
class MockAgentWorker(BaseAgentWorker):
    """Mock agent worker."""

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


class MockAgentWorkerWithMemory(MockAgentWorker):
    """Mock agent worker with memory."""

    def __init__(self, limit: int = 2):
        """Initialize."""
        self.limit = limit

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        # counter will be set to the last value in memory
        if len(task.memory.get_all()) > 0:
            start = int(cast(Any, task.memory.get_all()[-1].content))
        else:
            start = 0
        task.extra_state["counter"] = 0
        task.extra_state["start"] = start
        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            memory=task.memory,
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        task.extra_state["counter"] += 1
        counter = task.extra_state["counter"] + task.extra_state["start"]
        is_done = task.extra_state["counter"] >= self.limit

        new_steps = [step.get_next_step(step_id=str(uuid.uuid4()))]

        if is_done:
            task.memory.put(ChatMessage(role=MessageRole.USER, content=str(counter)))

        return TaskStepOutput(
            output=AgentChatResponse(response=f"counter: {counter}"),
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )


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


class MockFunctionCallingAgentWorker(MockAgentWorker):
    """Mock function calling agent worker."""

    counter = 0

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        task.extra_state["sources"] = []
        return super().initialize_step(task, **kwargs)

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        self.counter += 1
        task.extra_state["sources"].append(
            ToolOutput(
                tool_name=f"My Tool",
                content=f"This is the output of tool call {self.counter}",
                raw_input={},
                raw_output=f"This is the raw output of tool call {self.counter}",
            )
        )
        return super().run_step(step, task, **kwargs)


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


def test_agent_with_reset() -> None:
    """Test agents with reset."""
    # test e2e chat
    # NOTE: to use chat, output needs to be AgentChatResponse
    agent_runner = AgentRunner(agent_worker=MockAgentWorkerWithMemory(limit=10))
    for idx in range(4):
        if idx % 2 == 0:
            agent_runner.reset()

        response = agent_runner.chat("hello world")
        if idx % 2 == 0:
            assert str(response) == "counter: 10"
            assert len(agent_runner.state.task_dict) == 1
            assert len(agent_runner.memory.get_all()) == 1
        elif idx % 2 == 1:
            assert str(response) == "counter: 20"
            assert len(agent_runner.state.task_dict) == 2
            assert len(agent_runner.memory.get_all()) == 2


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
    from llama_index.core.agent import ReActAgent
    from llama_index.core.llms.mock import MockLLM

    llm = MockLLM()
    agent_runner = AgentRunner.from_llm(llm=llm)
    assert isinstance(agent_runner, ReActAgent)


def test_agent_dispatches_events() -> None:
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    num_steps = 3
    chat_message_input = "hello world"
    agent_runner = AgentRunner(agent_worker=MockAgentWorker(limit=num_steps))
    response = agent_runner.chat(chat_message_input)

    # Expect start / end event pair for agent chat, then a pair for each step run
    assert len(event_handler.events) == (2 + (2 * num_steps))

    # Check AgentChatWithStepStartEvent
    assert isinstance(event_handler.events[0], AgentChatWithStepStartEvent)
    assert event_handler.events[0].user_msg == chat_message_input

    # Check all AgentRunStepStartEvent
    for i in range(num_steps):
        run_step_start_idx = 2 * i + 1
        assert isinstance(
            event_handler.events[run_step_start_idx], AgentRunStepStartEvent
        )
        assert event_handler.events[run_step_start_idx].step is not None
        assert (
            event_handler.events[run_step_start_idx].step.task_id
            == event_handler.events[run_step_start_idx].task_id
        )
        assert event_handler.events[run_step_start_idx].input is None

    # Check all AgentRunStepEndEvent
    for i in range(num_steps):
        run_step_end_idx = 2 * i + 2
        assert isinstance(event_handler.events[run_step_end_idx], AgentRunStepEndEvent)
        assert (
            event_handler.events[run_step_end_idx].step_output.output.response
            == f"counter: {i + 1}"
        )
        assert (
            event_handler.events[run_step_end_idx].step_output.task_step
            == event_handler.events[run_step_end_idx - 1].step
        )
        assert len(event_handler.events[run_step_end_idx].step_output.next_steps) == 1

        # NOTE: MockAgentWorker generates a next step for the last step, so don't test this
        is_last_step = i == (num_steps - 1)
        if not is_last_step:
            assert (
                event_handler.events[run_step_end_idx].step_output.next_steps[0]
                == event_handler.events[run_step_end_idx + 1].step
            )

    # Check AgentChatWithStepEndEvent
    assert isinstance(event_handler.events[-1], AgentChatWithStepEndEvent)
    assert event_handler.events[-1].response == response


def test_agent_response_contains_full_source_history() -> None:
    # Test with a mock agent worker that records one source in each step
    num_steps = 4
    agent_runner = AgentRunner(
        agent_worker=MockFunctionCallingAgentWorker(limit=num_steps)
    )

    response = agent_runner.chat("hello world")
    assert len(response.sources) == num_steps

    # Expect that agent will collect all sources generated across all steps
    for step_idx in range(num_steps):
        assert (
            response.sources[step_idx].content
            == f"This is the output of tool call {step_idx + 1}"
        )


def test_agent_handles_nonexistent_source_history() -> None:
    num_steps = 4
    agent_runner = AgentRunner(agent_worker=MockAgentWorker(limit=num_steps))
    response = agent_runner.chat("hello world")
    assert len(response.sources) == 0
