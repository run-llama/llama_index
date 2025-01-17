from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Optional, Union

from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType, PromptDictType
from llama_index.core.tools import BaseTool, AsyncBaseTool, FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever
from llama_index.core.settings import Settings


def get_default_llm() -> LLM:
    return Settings.llm


class BaseWorkflowAgent(BaseModel, PromptMixin, ABC):
    """Base class for all agents, combining config and logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="The name of the agent")
    description: str = Field(
        description="The description of what the agent does and is responsible for"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="The system prompt for the agent"
    )
    tools: Optional[List[BaseTool]] = Field(
        default=None, description="The tools that the agent can use"
    )
    tool_retriever: Optional[ObjectRetriever] = Field(
        default=None,
        description="The tool retriever for the agent, can be provided instead of tools",
    )
    can_handoff_to: Optional[List[str]] = Field(
        default=None, description="The agent names that this agent can hand off to"
    )
    llm: LLM = Field(
        default_factory=get_default_llm, description="The LLM that the agent uses"
    )

    @field_validator("tools", mode="before")
    def validate_tools(
        cls, v: Optional[Sequence[Union[BaseTool, Callable]]]
    ) -> Optional[Sequence[BaseTool]]:
        """Validate tools.

        If tools are not of type BaseTool, they will be converted to FunctionTools.
        This assumes the inputs are tools or callable functions.
        """
        if v is None:
            return None

        validated_tools: List[BaseTool] = []
        for tool in v:
            if not isinstance(tool, BaseTool):
                validated_tools.append(FunctionTool.from_defaults(tool))
            else:
                validated_tools.append(tool)
        return validated_tools  # type: ignore[return-value]

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""

    @abstractmethod
    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the agent."""

    @abstractmethod
    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results."""

    @abstractmethod
    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the agent's execution."""
