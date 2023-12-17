from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union, cast

from llama_index.agent.types import (
    BaseAgent,
    BaseAgentStepEngine,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.llms.base import ChatMessage
from llama_index.llms.llm import LLM
from llama_index.llms.types import MessageRole
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.memory.types import BaseMemory


class BaseAgentRunner(BaseAgent):
    """Base agent runner."""


class TaskState(BaseModel):
    """Task state."""

    task: Task = Field(..., description="Task.")
    step_queue: Deque[TaskStep] = Field(
        default_factory=deque, description="Task step queue."
    )
    completed_steps: List[TaskStepOutput] = Field(
        default_factory=list, description="Completed step outputs."
    )


class AgentState(BaseModel):
    """Agent state."""

    task_dict: Dict[str, TaskState] = Field(
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


class AgentRunner(BaseAgentRunner):
    """agent runner."""

    # # TODO: implement this
    # step_executor: BaseAgentStepEngine
    # # stores tasks and their states
    # state: AgentState
    # # stores conversation history
    # # (currently assume agent runner handles a single message thread)
    # memory: BaseMemory
    # callback_manager: CallbackManager = Field(..., description="Callback manager.")

    # init_task_state_kwargs: Optional[dict] = Field(
    #     default_factory=dict, description="Initial task state kwargs."
    # )

    # class Config:
    #     arbitrary_types_allowed = True

    def __init__(
        self,
        step_executor: BaseAgentStepEngine,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        init_task_state_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize."""
        self.step_executor = step_executor
        self.state = state or AgentState()
        self.memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        self.callback_manager = callback_manager or CallbackManager([])
        self.init_task_state_kwargs = init_task_state_kwargs or {}

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    def reset(self) -> None:
        self.memory.reset()

    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""
        task = Task(
            input=input,
            step_queue=[],
            completed_steps=[],
            memory=self.memory,
            extra_state=self.init_task_state_kwargs,
            **kwargs,
        )
        # put input into memory
        self.memory.put(ChatMessage(content=input, role=MessageRole.USER))

        # get initial step from task, and put it in the step queue
        initial_step = self.step_executor.initialize_step(task)
        task_state = TaskState(
            task=task,
            step_queue=deque([initial_step]),
        )
        # add it to state
        self.state.task_dict[task.task_id] = task_state

        return task

    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """Delete task.

        NOTE: this will not delete any previous executions from memory.

        """
        self.state.task_dict.pop(task_id)

    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.step_executor.run_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.step_executor.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)
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
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible
        if mode == ChatResponseMode.WAIT:
            cur_step_output = await self.step_executor.arun_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = await self.step_executor.astream_step(
                step, task, **kwargs
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)
        return cur_step_output

    def run_step(
        self, task: Task, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step."""
        return self._run_step(task.task_id, step, mode=ChatResponseMode.WAIT, **kwargs)

    async def arun_step(
        self, task_id: str, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.WAIT, **kwargs
        )

    def stream_step(
        self, task: Task, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (stream)."""
        return self._run_step(
            task.task_id, step, mode=ChatResponseMode.STREAM, **kwargs
        )

    async def astream_step(
        self, task_id: str, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.STREAM, **kwargs
        )

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
            cur_step_output = self._run_step(task.task_id, mode=mode)

            if cur_step_output.is_last:
                # if cur_step_output.output is not AGENT_CHAT_RESPONSE_TYPE,
                # raise error
                if not isinstance(
                    cur_step_output.output,
                    (AgentChatResponse, StreamingAgentChatResponse),
                ):
                    raise ValueError(
                        "When `is_last` is True, cur_step_output.output must be "
                        f"AGENT_CHAT_RESPONSE_TYPE: {cur_step_output.output}"
                    )
                result_output = cur_step_output
                break

        # now that it is done, delete task
        self.delete_task(task.task_id)
        if result_output is None:
            raise ValueError("result_output is None")
        else:
            return cast(AGENT_CHAT_RESPONSE_TYPE, cur_step_output.output)

    async def _achat(
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
            cur_step_output = await self._arun_step(task.task_id, mode=mode)

            if cur_step_output.is_last:
                # if cur_step_output.output is not AGENT_CHAT_RESPONSE_TYPE,
                # raise error
                if not isinstance(
                    cur_step_output.output,
                    (AgentChatResponse, StreamingAgentChatResponse),
                ):
                    raise ValueError(
                        "When `is_last` is True, cur_step_output.output must be "
                        f"AGENT_CHAT_RESPONSE_TYPE: {cur_step_output.output}"
                    )
                result_output = cur_step_output
                break

        # now that it is done, delete task
        self.delete_task(task.task_id)
        if result_output is None:
            raise ValueError("result_output is None")
        else:
            return cast(AGENT_CHAT_RESPONSE_TYPE, cur_step_output.output)

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
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

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
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")
