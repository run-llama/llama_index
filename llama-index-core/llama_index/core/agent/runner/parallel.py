"""Agent executor."""

import asyncio
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union, cast

from llama_index.core.agent.runner.base import BaseAgentRunner
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory


class DAGTaskState(BaseModel):
    """DAG Task state."""

    task: Task = Field(..., description="Task.")
    root_step: TaskStep = Field(..., description="Root step.")
    step_queue: Deque[TaskStep] = Field(
        default_factory=deque, description="Task step queue."
    )
    completed_steps: List[TaskStepOutput] = Field(
        default_factory=list, description="Completed step outputs."
    )

    @property
    def task_id(self) -> str:
        """Task id."""
        return self.task.task_id


class DAGAgentState(BaseModel):
    """Agent state."""

    task_dict: Dict[str, DAGTaskState] = Field(
        default_factory=dict, description="Task dictionary."
    )

    def get_task(self, task_id: str) -> Task:
        """Get task state."""
        return self.task_dict[task_id].task

    def get_completed_steps(self, task_id: str) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.task_dict[task_id].completed_steps

    def get_step_queue(self, task_id: str) -> Deque[TaskStep]:
        """Get step queue."""
        return self.task_dict[task_id].step_queue


class ParallelAgentRunner(BaseAgentRunner):
    """
    Parallel agent runner.

    Executes steps in queue in parallel. Requires async support.

    """

    def __init__(
        self,
        agent_worker: BaseAgentWorker,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[DAGAgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        init_task_state_kwargs: Optional[dict] = None,
        delete_task_on_finish: bool = False,
    ) -> None:
        """Initialize."""
        self.memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        self.state = state or DAGAgentState()
        self.callback_manager = callback_manager or CallbackManager([])
        self.init_task_state_kwargs = init_task_state_kwargs or {}
        self.agent_worker = agent_worker
        self.delete_task_on_finish = delete_task_on_finish

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    def reset(self) -> None:
        self.memory.reset()

    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""
        task = Task(
            input=input,
            memory=self.memory,
            extra_state=self.init_task_state_kwargs,
            **kwargs,
        )
        # # put input into memory
        # self.memory.put(ChatMessage(content=input, role=MessageRole.USER))

        # add it to state
        # get initial step from task, and put it in the step queue
        initial_step = self.agent_worker.initialize_step(task)
        task_state = DAGTaskState(
            task=task,
            root_step=initial_step,
            step_queue=deque([initial_step]),
        )

        self.state.task_dict[task.task_id] = task_state

        return task

    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """
        Delete task.

        NOTE: this will not delete any previous executions from memory.

        """
        self.state.task_dict.pop(task_id)

    def get_completed_tasks(self, **kwargs: Any) -> List[Task]:
        """Get completed tasks."""
        task_states = list(self.state.task_dict.values())
        return [
            task_state.task
            for task_state in task_states
            if len(task_state.completed_steps) > 0
            and task_state.completed_steps[-1].is_last
        ]

    def get_task_output(self, task_id: str, **kwargs: Any) -> TaskStepOutput:
        """Get task output."""
        task_state = self.state.task_dict[task_id]
        if len(task_state.completed_steps) == 0:
            raise ValueError(f"No completed steps for task_id: {task_id}")
        return task_state.completed_steps[-1]

    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""
        task_states = list(self.state.task_dict.values())
        return [task_state.task for task_state in task_states]

    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""
        return self.state.get_task(task_id)

    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""
        return list(self.state.get_step_queue(task_id))

    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.state.get_completed_steps(task_id)

    def run_steps_in_queue(
        self,
        task_id: str,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> List[TaskStepOutput]:
        """
        Execute steps in queue.

        Run all steps in queue, clearing it out.

        Assume that all steps can be run in parallel.

        """
        return asyncio_run(self.arun_steps_in_queue(task_id, mode=mode, **kwargs))

    async def arun_steps_in_queue(
        self,
        task_id: str,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> List[TaskStepOutput]:
        """
        Execute all steps in queue.

        All steps in queue are assumed to be ready.

        """
        # first pop all steps from step_queue
        steps: List[TaskStep] = []
        while len(self.state.get_step_queue(task_id)) > 0:
            steps.append(self.state.get_step_queue(task_id).popleft())

        # take every item in the queue, and run it
        tasks = []
        for step in steps:
            tasks.append(self._arun_step(task_id, step=step, mode=mode, **kwargs))

        return await asyncio.gather(*tasks)

    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        task_queue = self.state.get_step_queue(task_id)
        step = step or task_queue.popleft()

        if not step.is_ready:
            raise ValueError(f"Step {step.step_id} is not ready")

        if mode == ChatResponseMode.WAIT:
            cur_step_output: TaskStepOutput = self.agent_worker.run_step(
                step, task, **kwargs
            )
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for next_step in cur_step_output.next_steps:
            if next_step.is_ready:
                task_queue.append(next_step)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        return cur_step_output

    async def _arun_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        task_queue = self.state.get_step_queue(task_id)
        step = step or task_queue.popleft()

        if not step.is_ready:
            raise ValueError(f"Step {step.step_id} is not ready")

        if mode == ChatResponseMode.WAIT:
            cur_step_output = await self.agent_worker.arun_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = await self.agent_worker.astream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for next_step in cur_step_output.next_steps:
            if next_step.is_ready:
                task_queue.append(next_step)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        return cur_step_output

    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""
        return self._run_step(task_id, step, mode=ChatResponseMode.WAIT, **kwargs)

    async def arun_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.WAIT, **kwargs
        )

    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""
        return self._run_step(task_id, step, mode=ChatResponseMode.STREAM, **kwargs)

    async def astream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.STREAM, **kwargs
        )

    def finalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""
        if step_output is None:
            step_output = self.state.get_completed_steps(task_id)[-1]
        if not step_output.is_last:
            raise ValueError(
                "finalize_response can only be called on the last step output"
            )

        if not isinstance(
            step_output.output,
            (AgentChatResponse, StreamingAgentChatResponse),
        ):
            raise ValueError(
                "When `is_last` is True, cur_step_output.output must be "
                f"AGENT_CHAT_RESPONSE_TYPE: {step_output.output}"
            )

        # finalize task
        self.agent_worker.finalize_task(self.state.get_task(task_id))

        if self.delete_task_on_finish:
            self.delete_task(task_id)

        return cast(AGENT_CHAT_RESPONSE_TYPE, step_output.output)

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_outputs = self.run_steps_in_queue(task.task_id, mode=mode)

            # check if a step output is_last
            is_last = any(
                cur_step_output.is_last for cur_step_output in cur_step_outputs
            )
            if is_last:
                if len(cur_step_outputs) > 1:
                    raise ValueError(
                        "More than one step output returned in final step."
                    )
                cur_step_output = cur_step_outputs[0]
                result_output = cur_step_output
                break

        return self.finalize_response(
            task.task_id,
            result_output,
        )

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            await self.memory.aset(chat_history)
        task = self.create_task(message)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_outputs = await self.arun_steps_in_queue(task.task_id, mode=mode)

            # check if a step output is_last
            is_last = any(
                cur_step_output.is_last for cur_step_output in cur_step_outputs
            )
            if is_last:
                if len(cur_step_outputs) > 1:
                    raise ValueError(
                        "More than one step output returned in final step."
                    )
                cur_step_output = cur_step_outputs[0]
                result_output = cur_step_output
                break

        return self.finalize_response(
            task.task_id,
            result_output,
        )

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            e.on_end(payload={EventPayload.RESPONSE: chat_response})

        return chat_response  # type: ignore

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )

            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response  # type: ignore

    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")
