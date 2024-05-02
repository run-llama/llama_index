"""
Hub is a central trustworthy that is aware of the existence of isolated apps, and that can reliably receive user queries and route them to the appropriate apps.
"""
from typing import Optional, Sequence, Callable


from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.llms import ChatMessage, MessageRole

from llama_index.core.llama_pack.base import BaseLlamaPack

from .planner import HubPlanner
from .tool_importer import ToolImporter
from .hub_operator import HubOperator


class Hub(BaseLlamaPack):
    """SecGPT Hub."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        tool_specs: Sequence[BaseToolSpec],
        llm: LLM = None,
        memory: BaseMemory = None,
        output_parser: Optional[ReActOutputParser] = None,
        verbose: bool = False,
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
        user_id: Optional[str] = "0",
    ) -> None:
        """Init params."""
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=[], llm=self.llm
        )
        self.output_parser = output_parser
        self.verbose = verbose
        self.handle_reasoning_failure_fn = handle_reasoning_failure_fn
        self.user_id = user_id

        self.planner = HubPlanner(self.llm)
        self.tool_importer = ToolImporter(tools, tool_specs)
        self.hub_operator = HubOperator(self.tool_importer, self.user_id)

    def chat(
        self,
        query: str,
    ) -> str:
        memory_content = self.memory.get()
        self.memory.put(ChatMessage(role=MessageRole.USER, content=(query)))
        tool_info = self.tool_importer.get_tool_info()
        plan = self.planner.plan_generate(query, tool_info, memory_content)
        response = self.hub_operator.run(query, plan)
        self.memory.put(ChatMessage(role=MessageRole.CHATBOT, content=(response)))

        return response
