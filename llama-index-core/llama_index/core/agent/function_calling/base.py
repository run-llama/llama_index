"""Function calling agent."""

from typing import (
    Any,
    List,
    Optional,
    Type,
)

from llama_index.agent.openai.step import OpenAIAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI


class FunctionCallingAgent(AgentRunner):
    """Function calling agent.

    Calls any LLM that supports function calling in a while loop until the task is complete.
    
    """

    # def __init__(
    #     self,
    #     tools: List[BaseTool],
    #     llm: OpenAI,
    #     memory: BaseMemory,
    #     prefix_messages: List[ChatMessage],
    #     verbose: bool = False,
    #     max_function_calls: int = 5,
    #     default_tool_choice: str = "auto",
    # )

