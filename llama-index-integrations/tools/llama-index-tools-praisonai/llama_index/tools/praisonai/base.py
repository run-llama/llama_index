"""PraisonAI tool spec for LlamaIndex."""

from typing import List
import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class PraisonAIToolSpec(BaseToolSpec):
    """
    PraisonAI tool spec for running multi-agent workflows.

    This tool allows LlamaIndex agents to interact with a PraisonAI server
    to execute multi-agent workflows.
    """

    spec_functions = ["run_workflow", "run_agent", "list_agents"]

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        timeout: int = 300,
    ) -> None:
        """
        Initialize the PraisonAI tool spec.

        Args:
            api_url: The URL of the PraisonAI server.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url
        self.timeout = timeout

    def run_workflow(self, query: str) -> str:
        """
        Run a query through the PraisonAI multi-agent workflow.

        Args:
            query: The query to send to the agents.

        Returns:
            The response from the agents.
        """
        response = httpx.post(
            f"{self.api_url}/agents",
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def run_agent(self, query: str, agent: str) -> str:
        """
        Run a query through a specific PraisonAI agent.

        Args:
            query: The query to send to the agent.
            agent: The name of the agent to use.

        Returns:
            The response from the agent.
        """
        response = httpx.post(
            f"{self.api_url}/agents/{agent}",
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def list_agents(self) -> List[str]:
        """
        List available agents on the PraisonAI server.

        Returns:
            A list of agent names.
        """
        response = httpx.get(
            f"{self.api_url}/agents/list",
            timeout=30,
        )
        response.raise_for_status()
        agents = response.json().get("agents", [])
        return [a.get("name", "") for a in agents]
