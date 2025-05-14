import os
from abc import abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union, cast

from llama_index.core.agent.types import (
    BaseAgent,
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.async_utils import asyncio_run, run_jobs
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
from llama_index.core.tools.types import BaseTool
from llama_index.core.instrumentation.events.agent import (
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class BaseAgentRunner(BaseAgent):
    """Base agent runner."""

    @abstractmethod
    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""

    @abstractmethod
    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """
        Delete task.

        NOTE: this will not delete any previous executions from memory.

        """

    @abstractmethod
    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""

    @abstractmethod
    def get_completed_tasks(self, **kwargs: Any) -> List[Task]:
        """Get completed tasks."""

    @abstractmethod
    def get_task_output(self, task_id: str, **kwargs: Any) -> TaskStepOutput:
        """Get task output."""

    @abstractmethod
    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""

    @abstractmethod
    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""

    @abstractmethod
    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""

    def get_completed_step(
        self, task_id: str, step_id: str, **kwargs: Any
    ) -> TaskStepOutput:
        """Get completed step."""
        # call get_completed_steps, and then find the right task
        completed_steps = self.get_completed_steps(task_id, **kwargs)
        for step_output in completed_steps:
            if step_output.task_step.step_id == step_id:
                return step_output
        raise ValueError(f"Could not find step_id: {step_id}")

    @abstractmethod
    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""

    @abstractmethod
    async def arun_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async)."""

    @abstractmethod
    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""

    @abstractmethod
    async def astream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async stream)."""

    @abstractmethod
    def finalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""

    async def afinalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""
        return self.finalize_response(task_id, step_output)

    @abstractmethod
    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")


def validate_step_from_args(
    task_id: str, input: Optional[str] = None, step: Optional[Any] = None, **kwargs: Any
) -> Optional[TaskStep]:
    """Validate step from args."""
    if step is not None:
        if input is not None:
            raise ValueError("Cannot specify both `step` and `input`")
        if not isinstance(step, TaskStep):
            raise ValueError(f"step must be TaskStep: {step}")
        return step
    else:
        return None


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

    def reset(self) -> None:
        """Reset."""
        self.task_dict = {}


class AgentRunner(BaseAgentRunner):
    """
    Agent runner.

    Top-level agent orchestrator that can create tasks, run each step in a task,
    or run a task e2e. Stores state and keeps track of tasks.

    Args:
        agent_worker (BaseAgentWorker): step executor
        chat_history (Optional[List[ChatMessage]], optional): chat history. Defaults to None.
        state (Optional[AgentState], optional): agent state. Defaults to None.
        memory (Optional[BaseMemory], optional): memory. Defaults to None.
        llm (Optional[LLM], optional): LLM. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): callback manager. Defaults to None.
        init_task_state_kwargs (Optional[dict], optional): init task state kwargs. Defaults to None.

    """

    # # TODO: implement this in Pydantic

    def __init__(
        self,
        agent_worker: BaseAgentWorker,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        init_task_state_kwargs: Optional[dict] = None,
        delete_task_on_finish: bool = False,
        default_tool_choice: str = "auto",
        verbose: bool = False,
    ) -> None:
        """Initialize."""
        self.agent_worker = agent_worker
        self.state = state or AgentState()
        self.memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)

        # get and set callback manager
        if callback_manager is not None:
            self.agent_worker.set_callback_manager(callback_manager)
            self.callback_manager = callback_manager
        else:
            # TODO: This is *temporary*
            # Stopgap before having a callback on the BaseAgentWorker interface.
            # Doing that requires a bit more refactoring to make sure existing code
            # doesn't break.
            if hasattr(self.agent_worker, "callback_manager"):
                self.callback_manager = (
                    self.agent_worker.callback_manager or CallbackManager()
                )
            else:
                self.callback_manager = CallbackManager()
        self.init_task_state_kwargs = init_task_state_kwargs or {}
        self.delete_task_on_finish = delete_task_on_finish
        self.default_tool_choice = default_tool_choice
        self.verbose = verbose

    @staticmethod
    def from_llm(
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "AgentRunner":
        from llama_index.core.agent import ReActAgent

        if os.getenv("IS_TESTING"):
            return ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                **kwargs,
            )

        try:
            from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
            from llama_index.llms.openai.utils import (
                is_function_calling_model,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "`llama-index-llms-openai` package not found. Please "
                "install by running `pip install llama-index-llms-openai`."
            )

        if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
            from llama_index.agent.openai import OpenAIAgent  # pants: no-infer-dep

            return OpenAIAgent.from_tools(
                tools=tools,
                llm=llm,
                **kwargs,
            )
        else:
            return ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                **kwargs,
            )

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    def reset(self) -> None:
        self.memory.reset()
        self.state.reset()

    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""
        if not self.init_task_state_kwargs:
            extra_state = kwargs.pop("extra_state", {})
        else:
            if "extra_state" in kwargs:
                raise ValueError(
                    "Cannot specify both `extra_state` and `init_task_state_kwargs`"
                )
            else:
                extra_state = self.init_task_state_kwargs

        callback_manager = kwargs.pop("callback_manager", self.callback_manager)
        task = Task(
            input=input,
            memory=self.memory,
            extra_state=extra_state,
            callback_manager=callback_manager,
            **kwargs,
        )
        # # put input into memory
        # self.memory.put(ChatMessage(content=input, role=MessageRole.USER))

        # get initial step from task, and put it in the step queue
        initial_step = self.agent_worker.initialize_step(task)
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
        """
        Delete task.

        NOTE: this will not delete any previous executions from memory.

        """
        self.state.task_dict.pop(task_id)

    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""
        return [task_state.task for task_state in self.state.task_dict.values()]

    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""
        return self.state.get_task(task_id)

    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""
        return list(self.state.get_step_queue(task_id))

    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.state.get_completed_steps(task_id)

    def get_task_output(self, task_id: str, **kwargs: Any) -> TaskStepOutput:
        """Get task output."""
        completed_steps = self.get_completed_steps(task_id)
        if len(completed_steps) == 0:
            raise ValueError(f"No completed steps for task_id: {task_id}")
        return completed_steps[-1]

    def get_completed_tasks(self, **kwargs: Any) -> List[Task]:
        """Get completed tasks."""
        task_states = list(self.state.task_dict.values())
        completed_tasks = []
        for task_state in task_states:
            completed_steps = self.get_completed_steps(task_state.task.task_id)
            if len(completed_steps) > 0 and completed_steps[-1].is_last:
                completed_tasks.append(task_state.task)

        return completed_tasks

    @dispatcher.span
    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()
        if input is not None:
            step.input = input

        dispatcher.event(
            AgentRunStepStartEvent(task_id=task_id, step=step, input=input)
        )

        if self.verbose:
            print(f"> Running step {step.step_id}. Step input: {step.input}")

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
        return cur_step_output

    @dispatcher.span
    async def _arun_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        dispatcher.event(
            AgentRunStepStartEvent(task_id=task_id, step=step, input=input)
        )
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()
        if input is not None:
            step.input = input

        if self.verbose:
            print(f"> Running step {step.step_id}. Step input: {step.input}")

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible
        if mode == ChatResponseMode.WAIT:
            cur_step_output = await self.agent_worker.arun_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = await self.agent_worker.astream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
        return cur_step_output

    @dispatcher.span
    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return self._run_step(
            task_id, step, input=input, mode=ChatResponseMode.WAIT, **kwargs
        )

    @dispatcher.span
    async def arun_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return await self._arun_step(
            task_id, step, input=input, mode=ChatResponseMode.WAIT, **kwargs
        )

    @dispatcher.span
    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return self._run_step(
            task_id, step, input=input, mode=ChatResponseMode.STREAM, **kwargs
        )

    @dispatcher.span
    async def astream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return await self._arun_step(
            task_id, step, input=input, mode=ChatResponseMode.STREAM, **kwargs
        )

    @dispatcher.span
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

        # Attach all sources generated across all steps
        step_output.output.sources = self.get_task(task_id).extra_state.get(
            "sources", []
        )
        step_output.output.set_source_nodes()

        return cast(AGENT_CHAT_RESPONSE_TYPE, step_output.output)

    @dispatcher.span
    async def afinalize_response(
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
        await self.agent_worker.afinalize_task(self.state.get_task(task_id))

        if self.delete_task_on_finish:
            self.delete_task(task_id)

        # Attach all sources generated across all steps
        step_output.output.sources = self.get_task(task_id).extra_state.get(
            "sources", []
        )
        step_output.output.set_source_nodes()

        return cast(AGENT_CHAT_RESPONSE_TYPE, step_output.output)

    @dispatcher.span
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
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self._run_step(
                task.task_id, mode=mode, tool_choice=tool_choice
            )

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"

        result = self.finalize_response(
            task.task_id,
            result_output,
        )
        dispatcher.event(AgentChatWithStepEndEvent(response=result))
        return result

    @dispatcher.span
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
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = await self._arun_step(
                task.task_id, mode=mode, tool_choice=tool_choice
            )

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"

        result = await self.afinalize_response(
            task.task_id,
            result_output,
        )
        dispatcher.event(AgentChatWithStepEndEvent(response=result))
        return result

    @dispatcher.span
    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> AgentChatResponse:
        # override tool choice is provided as input.
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message=message,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT,
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @dispatcher.span
    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> AgentChatResponse:
        # override tool choice is provided as input.
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message=message,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT,
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @dispatcher.span
    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> StreamingAgentChatResponse:
        # override tool choice is provided as input.
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse) or (
                isinstance(chat_response, AgentChatResponse)
                and chat_response.is_dummy_stream
            )
            e.on_end(payload={EventPayload.RESPONSE: chat_response})

        return chat_response  # type: ignore

    @dispatcher.span
    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> StreamingAgentChatResponse:
        # override tool choice is provided as input.
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse) or (
                isinstance(chat_response, AgentChatResponse)
                and chat_response.is_dummy_stream
            )
            e.on_end(payload={EventPayload.RESPONSE: chat_response})

        return chat_response  # type: ignore

    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")


class BasePlanningAgentRunner(AgentRunner):
    @abstractmethod
    def create_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan. Returns the plan_id."""
        ...

    @abstractmethod
    def get_next_tasks(self, plan_id: str, **kwargs: Any) -> List[str]:
        """Get next task ids for a given plan."""
        ...

    @abstractmethod
    def mark_task_complete(self, plan_id: str, task_id: str, **kwargs: Any) -> None:
        """Mark task complete for a given plan."""
        ...

    @abstractmethod
    def refine_plan(self, input: str, plan_id: str, **kwargs: Any) -> None:
        """Refine plan."""
        ...

    @abstractmethod
    def run_task(self, task_id: str, **kwargs: Any) -> AGENT_CHAT_RESPONSE_TYPE:
        """Run task."""
        ...

    async def acreate_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan (async). Returns the plan_id."""
        return self.create_plan(input, **kwargs)

    async def arefine_plan(self, input: str, plan_id: str, **kwargs: Any) -> None:
        """Refine plan (async)."""
        return self.refine_plan(input, plan_id, **kwargs)

    async def arun_task(self, task_id: str, **kwargs: Any) -> AGENT_CHAT_RESPONSE_TYPE:
        """Run task (async)."""
        return self.run_task(task_id, **kwargs)

    @dispatcher.span
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

        # create initial set of tasks
        plan_id = self.create_plan(message)

        results = []
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            # EXIT CONDITION: check if all sub-tasks are completed
            next_task_ids = self.get_next_tasks(plan_id)
            if len(next_task_ids) == 0:
                break

            jobs = [
                self.arun_task(sub_task_id, mode=mode, tool_choice=tool_choice)
                for sub_task_id in next_task_ids
            ]
            results = asyncio_run(run_jobs(jobs, workers=len(jobs)))

            for sub_task_id in next_task_ids:
                self.mark_task_complete(plan_id, sub_task_id)

            # EXIT CONDITION: check if all sub-tasks are completed now
            # LLMs have a tendency to add more tasks, so we end if there are no more tasks
            # next_sub_tasks = self.state.get_next_sub_tasks(plan_id)
            # if len(next_sub_tasks) == 0:
            #    break

            # refine the plan
            self.refine_plan(message, plan_id)

        dispatcher.event(
            AgentChatWithStepEndEvent(
                response=results[-1] if len(results) > 0 else None
            )
        )
        return results[-1]

    @dispatcher.span
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

        # create initial set of tasks
        plan_id = self.create_plan(message)

        results = []
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            # EXIT CONDITION: check if all sub-tasks are completed
            next_task_ids = self.get_next_tasks(plan_id)
            if len(next_task_ids) == 0:
                break

            jobs = [
                self.arun_task(sub_task_id, mode=mode, tool_choice=tool_choice)
                for sub_task_id in next_task_ids
            ]
            results = await run_jobs(jobs, workers=len(jobs))

            for sub_task_id in next_task_ids:
                self.mark_task_complete(plan_id, sub_task_id)

            # EXIT CONDITION: check if all sub-tasks are completed now
            # LLMs have a tendency to add more tasks, so we end if there are no more tasks
            # next_sub_tasks = self.state.get_next_sub_tasks(plan_id)
            # if len(next_sub_tasks) == 0:
            #    break

            # refine the plan
            await self.arefine_plan(message, plan_id)

        dispatcher.event(
            AgentChatWithStepEndEvent(
                response=results[-1] if len(results) > 0 else None
            )
        )
        return results[-1]
