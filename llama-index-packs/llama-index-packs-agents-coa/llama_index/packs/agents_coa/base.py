"""LLM Compiler agent pack."""

from typing import Any, Dict, List, Optional

from llama_index.core.agent import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.settings import Settings
from llama_index.core.tools.types import BaseTool

from .step import CoAAgentWorker


class CoAAgentPack(BaseLlamaPack):
    """
    Chain-of-abstraction Agent Pack.

    Args:
        tools (List[BaseTool]): List of tools to use.
        llm (Optional[LLM]): LLM to use. Defaults to gpt-4.

    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        agent_worker_kwargs: Optional[Dict[str, Any]] = None,
        agent_runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Init params."""
        self.llm = llm or Settings.llm
        self.callback_manager = callback_manager or self.llm.callback_manager
        self.agent_worker = CoAAgentWorker.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            callback_manager=self.callback_manager,
            **(agent_worker_kwargs or {}),
        )
        self.agent = AgentRunner(
            self.agent_worker,
            callback_manager=self.callback_manager,
            **(agent_runner_kwargs or {}),
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self.llm,
            "callback_manager": self.callback_manager,
            "agent_worker": self.agent_worker,
            "agent": self.agent,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.agent.chat(*args, **kwargs)
