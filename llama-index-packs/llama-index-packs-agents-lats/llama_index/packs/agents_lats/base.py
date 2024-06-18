from typing import List, Any, Dict, Optional

from llama_index.core.agent import AgentRunner
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool

from llama_index.packs.agents_lats.step import LATSAgentWorker


class LATSPack(BaseLlamaPack):
    """Pack for running the LATS agent."""

    def __init__(
        self, tools: List[BaseTool], llm: Optional[LLM] = None, **kwargs: Any
    ) -> None:
        """Init params."""
        self.agent_worker = LATSAgentWorker(tools=tools, llm=llm, **kwargs)
        self.agent = AgentRunner(self.agent_worker)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "agent_worker": self.agent_worker,
            "agent": self.agent,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run."""
        return self.agent.chat(*args, **kwargs)
