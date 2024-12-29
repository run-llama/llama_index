from enum import Enum
from typing import List, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.llms import LLM
from llama_index.core.objects import ObjectRetriever
from llama_index.core.tools import BaseTool


class AgentMode(str, Enum):
    """Agent mode."""

    DEFAULT = "default"
    REACT = "react"
    FUNCTION = "function"


class AgentConfig(BaseModel):
    """Configuration for a single agent in the multi-agent system."""

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
    mode: AgentMode = AgentMode.DEFAULT

    def get_mode(self) -> AgentMode:
        """Resolve the mode of the agent."""
        if self.mode == AgentMode.DEFAULT:
            return (
                AgentMode.FUNCTION
                if self.llm.metadata.is_function_calling_model
                else AgentMode.REACT
            )

        return self.mode
