"""LlamaPack class."""

from typing import Any, Dict

from llama_index.core.llama_pack.base import BaseLlamaPack

# backwards compatibility
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.async_utils import asyncio_run


class GmailOpenAIAgentPack(BaseLlamaPack):
    def __init__(self, gmail_tool_kwargs: Dict[str, Any]) -> None:
        """Init params."""
        try:
            from llama_index.tools.google import GmailToolSpec
        except ImportError:
            raise ImportError("llama_hub not installed.")

        self.tool_spec = GmailToolSpec(**gmail_tool_kwargs)
        self.agent = FunctionAgent(
            tools=self.tool_spec.to_tool_list(),
            llm=OpenAI(model="gpt-4.1"),
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"gmail_tool": self.tool_spec, "agent": self.agent}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return asyncio_run(self.arun(*args, **kwargs))

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline asynchronously."""
        return await self.agent.run(*args, **kwargs)
