"""Custom agent worker."""

import uuid
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from llama_index.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.tools.types import AsyncBaseTool

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


class CustomSimpleAgentWorker(BaseModel, BaseAgentWorker):
    """Custom simple agent worker.

    This is "simple" in the sense that some of the scaffolding is setup already.
    Assumptions:
    - assumes that the agent has tools, llm, callback manager, and tool retriever
    - has a `from_tools` convenience function
    - assumes that the agent is sequential, and doesn't take in any additional
    intermediate inputs.

    Args:
        tools (Sequence[BaseTool]): Tools to use for reasoning
        llm (LLM): LLM to use
        callback_manager (CallbackManager): Callback manager
        tool_retriever (Optional[ObjectRetriever[BaseTool]]): Tool retriever
        verbose (bool): Whether to print out reasoning steps

    """

    tools: Sequence[BaseTool] = Field(..., description="Tools to use for reasoning")
    llm: LLM = Field(..., description="LLM to use")
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    tool_retriever: Optional[ObjectRetriever[BaseTool]] = Field(
        default=None, description="Tool retriever"
    )
    verbose: bool = Field(False, description="Whether to print out reasoning steps")

    _get_tools: Callable[[str], Sequence[BaseTool]] = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []

        super().__init__(
            tools=tools,
            llm=llm,
            callback_manager=callback_manager,
            tool_retriever=tool_retriever,
            verbose=verbose,
        )

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CustomSimpleAgentWorker":
        """Convenience constructor method from set of of BaseTools (Optional)."""
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @abstractmethod
    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize initial state
        initial_state = {
            "sources": sources,
            "memory": new_memory,
        }

        step_state = self._initialize_state(task, **kwargs)
        # if intersecting keys, error
        if set(step_state.keys()).intersection(set(initial_state.keys())):
            raise ValueError(
                f"Step state keys {step_state.keys()} and initial state keys {initial_state.keys()} intersect."
                f"*NOTE*: initial state keys {initial_state.keys()} are reserved."
            )
        step_state.update(initial_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state=step_state,
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [adapt_to_async_tool(t) for t in self._get_tools(input)]

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    @abstractmethod
    def _run_step(
        self,
        state: Dict[str, Any],
        task: Task,
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """

    async def _arun_step(
        self,
        state: Dict[str, Any],
        task: Task,
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step (async).

        Can override this method if you want to run the step asynchronously.

        Returns:
            Tuple of (agent_response, is_done)

        """
        raise NotImplementedError(
            "This agent does not support async." "Please implement _arun_step."
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        agent_response, is_done = self._run_step(step.step_state, task)
        response = self._get_task_step_response(agent_response, step, is_done)
        # sync step state with task state
        task.extra_state.update(step.step_state)
        return response

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        agent_response, is_done = await self._arun_step(step.step_state, task)
        response = self._get_task_step_response(agent_response, step, is_done)
        task.extra_state.update(step.step_state)
        return response

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError("This agent does not support streaming.")

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError("This agent does not support streaming.")

    @abstractmethod
    def _finalize_task(self, state: Dict[str, Any], **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed.

        State is all the step states.

        """

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(task.memory.get() + task.extra_state["memory"].get_all())
        # reset new memory
        task.extra_state["memory"].reset()
        self._finalize_task(task.extra_state, **kwargs)
