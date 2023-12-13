from typing import Any, List, Optional, Union, cast

from llama_index.agent.types import BaseAgent
from llama_index.agent.v1.schema import (
    AgentState,
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
from llama_index.llms.base import LLM, ChatMessage, MessageRole
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.response.schema import Response

# def _add_initial_step(step_queue: Deque[TaskStep], task: Task):
#     """Add initial step."""
#     step_queue.append(
#         TaskStep(
#             task_id=task.task_id,
#             step_id=0,
#             input=task.input,
#             memory=task.memory,
#         )
#     )


class AgentEngine(BaseModel, BaseAgent):
    """Agent engine."""

    # TODO: implement this
    step_executor: BaseAgentStepEngine
    # stores tasks and their states
    state: AgentState
    # stores conversation history
    # (currently assume agent engine handles a single message thread)
    memory: BaseMemory
    callback_manager: CallbackManager = Field(..., description="Callback manager.")

    init_task_state_kwargs: Optional[dict] = Field(
        default_factory=dict, description="Initial task state kwargs."
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        step_executor: BaseAgentStepEngine,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        init_task_state_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize."""
        memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        state = state or AgentState()
        callback_manager = callback_manager or CallbackManager([])
        init_task_state_kwargs = init_task_state_kwargs or {}
        super().__init__(
            step_executor=step_executor,
            state=state,
            memory=memory,
            callback_manager=callback_manager,
            init_task_state_kwargs=init_task_state_kwargs,
        )

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
            **kwargs
        )
        # add it to state
        self.state.task_dict[task.task_id] = task
        return task

    def run_step(
        self, task_id: str, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step = step or task.step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        cur_step_output = self.step_executor.run_step(step, task, **kwargs)
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        task.step_queue.extend(next_steps)
        return cur_step_output

    async def arun_step(
        self, task_id: str, step: Optional[TaskStep] = None, **kwargs: Any
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step = step or task.step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        cur_step_output = await self.step_executor.arun_step(step, task, **kwargs)
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        task.step_queue.extend(next_steps)
        return cur_step_output

    # def _query(self, query: QueryBundle) -> Response:
    #     """Run an e2e execution of a query."""
    #     task = self.create_task(query)

    #     initial_step = self.step_executor.initialize_step(task)
    #     task.step_queue.append(initial_step)

    #     result_output = None
    #     while True:
    #         # pass step queue in as argument, assume step executor is stateless
    #         cur_step_output = self.run_step(task.task_id)

    #         if cur_step_output.is_last:
    #             result_output = cur_step_output
    #             break
    #     return Response(response=str(result_output))

    def initialize_state_and_task(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Task:
        # init chat with memory
        if chat_history is not None:
            self.memory.set(chat_history)
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))

        # TODO: include chat history
        task = self.create_task(message)
        initial_step = self.step_executor.initialize_step(task)
        task.step_queue.append(initial_step)
        return task

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        task = self.initialize_state_and_task(message, chat_history)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self.run_step(task.task_id)

            if cur_step_output.is_last:
                result_output = cur_step_output
                break
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
        task = self.initialize_state_and_task(message, chat_history)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = await self.arun_step(task.task_id)

            if cur_step_output.is_last:
                result_output = cur_step_output
                break
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
