"""Base agent type."""

import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.memory.types import BaseMemory
from llama_index.core.prompts.mixin import PromptDictType, PromptMixin, PromptMixinType
from llama_index.core.schema import QueryBundle


class BaseAgent(BaseChatEngine, BaseQueryEngine):
    """Base Agent."""

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct agent does not explicitly specify prompts, would need a
        # refactor to expose those prompts
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    # ===== Query Engine Interface =====
    @trace_method("query")
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(
            response=str(agent_response), source_nodes=agent_response.source_nodes
        )

    @trace_method("query")
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(
            response=str(agent_response), source_nodes=agent_response.source_nodes
        )

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")


class TaskStep(BaseModel):
    """Agent task step.

    Represents a single input step within the execution run ("Task") of an agent
    given a user input.

    The output is returned as a `TaskStepOutput`.

    """

    task_id: str = Field(..., diescription="Task ID")
    step_id: str = Field(..., description="Step ID")
    input: Optional[str] = Field(default=None, description="User input")
    # memory: BaseMemory = Field(
    #     ..., type=BaseMemory, description="Conversational Memory"
    # )
    step_state: Dict[str, Any] = Field(
        default_factory=dict, description="Additional state for a given step."
    )

    # NOTE: the state below may change throughout the course of execution
    # this tracks the relationships to other steps
    next_steps: Dict[str, "TaskStep"] = Field(
        default_factory=dict, description="Next steps to be executed."
    )
    prev_steps: Dict[str, "TaskStep"] = Field(
        default_factory=dict,
        description="Previous steps that were dependencies for this step.",
    )
    is_ready: bool = Field(
        default=True, description="Is this step ready to be executed?"
    )

    def get_next_step(
        self,
        step_id: str,
        input: Optional[str] = None,
        step_state: Optional[Dict[str, Any]] = None,
    ) -> "TaskStep":
        """Convenience function to get next step.

        Preserve task_id, memory, step_state.

        """
        return TaskStep(
            task_id=self.task_id,
            step_id=step_id,
            input=input,
            # memory=self.memory,
            step_state=step_state or self.step_state,
        )

    def link_step(
        self,
        next_step: "TaskStep",
    ) -> None:
        """Link to next step.

        Add link from this step to next, and from next step to current.

        """
        self.next_steps[next_step.step_id] = next_step
        next_step.prev_steps[self.step_id] = self


class TaskStepOutput(BaseModel):
    """Agent task step output."""

    output: Any = Field(..., description="Task step output")
    task_step: TaskStep = Field(..., description="Task step input")
    next_steps: List[TaskStep] = Field(..., description="Next steps to be executed.")
    is_last: bool = Field(default=False, description="Is this the last step?")

    def __str__(self) -> str:
        """String representation."""
        return str(self.output)


class Task(BaseModel):
    """Agent Task.

    Represents a "run" of an agent given a user input.

    """

    class Config:
        arbitrary_types_allowed = True

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), type=str, description="Task ID"
    )
    input: str = Field(..., type=str, description="User input")

    # NOTE: this is state that may be modified throughout the course of execution of the task
    memory: BaseMemory = Field(
        ...,
        type=BaseMemory,
        description=(
            "Conversational Memory. Maintains state before execution of this task."
        ),
    )

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager,
        exclude=True,
        description="Callback manager for the task.",
    )

    extra_state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional user-specified state for a given task. "
            "Can be modified throughout the execution of a task."
        ),
    )


class BaseAgentWorker(PromptMixin):
    """Base agent worker."""

    class Config:
        arbitrary_types_allowed = True

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct agent does not explicitly specify prompts, would need a
        # refactor to expose those prompts
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    @abstractmethod
    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""

    @abstractmethod
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""

    @abstractmethod
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        raise NotImplementedError

    @abstractmethod
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        raise NotImplementedError

    @abstractmethod
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError

    @abstractmethod
    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: make this abstractmethod (right now will break some agent impls)
