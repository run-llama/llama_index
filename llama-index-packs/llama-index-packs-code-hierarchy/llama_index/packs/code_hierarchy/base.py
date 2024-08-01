from typing import Any, Dict, List

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.schema import BaseNode
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool


class CodeHierarchyAgentPack(BaseLlamaPack):
    """Code hierarchy agent pack."""

    def __init__(self, split_nodes: List[BaseNode], llm: OpenAI, verbose: bool = True):
        """Initialize the code hierarchy agent pack."""
        from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine

        self.query_engine = CodeHierarchyKeywordQueryEngine(
            nodes=split_nodes,
        )

        self.tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="code_search",
            description="Search the code hierarchy for a specific code element, using keywords or IDs.",
        )

        self.agent = OpenAIAgent.from_tools(
            tools=[self.tool],
            llm=llm,
            system_prompt=self.query_engine.get_tool_instructions(),
            verbose=verbose,
        )

    def get_modules(self) -> Dict[str, Any]:
        return {
            "query_engine": self.query_engine,
            "tool": self.tool,
            "agent": self.agent,
        }

    def run(self, user_message: str) -> str:
        """Run the agent on the user message."""
        return str(self.agent.chat(user_message))
