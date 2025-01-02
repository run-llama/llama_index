from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import BaseTool, AsyncBaseTool
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever


class BaseWorkflowAgent(BaseModel, ABC):
    """Base class for all agents, combining config and logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: Optional[str] = None
    tools: Optional[List[BaseTool]] = None
    tool_retriever: Optional[ObjectRetriever] = None
    can_handoff_to: Optional[List[str]] = Field(default=None)
    handoff_prompt_template: Optional[str] = None
    llm: Optional[LLM] = None
    is_entrypoint_agent: bool = False

    @abstractmethod
    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
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
